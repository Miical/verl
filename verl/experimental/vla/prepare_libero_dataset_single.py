# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess LIBERO task-suite metadata to parquet format.

New feature:
- Support generating a dataset for only one task via --single_task_id
"""

import argparse
import os
import random

import numpy as np
import torch
from datasets import Dataset
from libero.libero import get_libero_path
from libero.libero.benchmark import Benchmark, get_benchmark, get_benchmark_dict


def patched_get_task_init_states(self, i):
    init_states_path = os.path.join(
        get_libero_path("init_states"),
        self.tasks[i].problem_folder,
        self.tasks[i].init_states_file,
    )
    init_states = torch.load(init_states_path, weights_only=False)
    return init_states


Benchmark.get_task_init_states = patched_get_task_init_states


def compute_total_num_group_envs(task_suite: Benchmark):
    total_num_group_envs = 0
    trial_id_bins = []
    for task_id in range(task_suite.get_num_tasks()):
        task_num_trials = len(task_suite.get_task_init_states(task_id))
        trial_id_bins.append(task_num_trials)
        total_num_group_envs += task_num_trials

    cumsum_trial_id_bins = np.cumsum(trial_id_bins)
    return total_num_group_envs, cumsum_trial_id_bins


def build_dataset_for_suite(
    task_suite_name: str,
    local_save_dir: str,
    single_task_id: int | None = None,
    max_train_per_task: int = 40,
    max_test_per_task: int = 10,
):
    task_suite = get_benchmark(task_suite_name)()
    total_num_group_envs, cumsum_trial_id_bins = compute_total_num_group_envs(task_suite)

    print(f"\n[Suite: {task_suite_name}]")
    print(f"Total number of group envs: {total_num_group_envs}")
    print(f"Cumsum trial id bins: {cumsum_trial_id_bins}")

    num_tasks = task_suite.get_num_tasks()
    all_task_ids = list(range(num_tasks))

    def get_state_ids_for_task(task_id: int):
        start_id = 0 if task_id == 0 else cumsum_trial_id_bins[task_id - 1]
        end_id = cumsum_trial_id_bins[task_id]
        return list(range(start_id, end_id))

    if single_task_id is not None:
        if single_task_id < 0 or single_task_id >= num_tasks:
            raise ValueError(
                f"single_task_id={single_task_id} is out of range for suite {task_suite_name}. "
                f"Valid range: [0, {num_tasks - 1}]"
            )
        train_task_ids = [single_task_id]
        ood_test_task_ids = []
        print("\n[Single Task Mode]")
        print(f"Only generating task_id={single_task_id}")
        print(f"Task language: {task_suite.get_task(single_task_id).language}")
    else:
        if len(all_task_ids) > 1:
            train_task_num = max(1, len(all_task_ids) - 1)
            train_task_ids = sorted(random.sample(all_task_ids, train_task_num))
            ood_test_task_ids = sorted(list(set(all_task_ids) - set(train_task_ids)))
        else:
            train_task_ids = all_task_ids
            ood_test_task_ids = []

    print("\n[Data Split Plan]")
    print(f"Training Task IDs: {train_task_ids}")
    print(f"OOD Test Task IDs: {ood_test_task_ids}")

    train_metadata = []
    test_metadata = []

    # Train split + ID test split
    for task_id in train_task_ids:
        all_trials = get_state_ids_for_task(task_id)
        random.shuffle(all_trials)

        train_count = int(len(all_trials) * 0.8)
        train_count = min(train_count, max_train_per_task)

        selected_train_trials = all_trials[:train_count]
        selected_id_test_trials = all_trials[train_count:]

        for state_id in selected_train_trials:
            train_metadata.append(
                {
                    "task_id": task_id,
                    "state_id": state_id,
                    "data_source": "train",
                }
            )

        for state_id in selected_id_test_trials[:max_test_per_task]:
            test_metadata.append(
                {
                    "task_id": task_id,
                    "state_id": state_id,
                    "data_source": "test_in_distribution",
                }
            )

    # OOD split: only used when not in single-task mode
    for ood_task_id in ood_test_task_ids:
        ood_all_trials = get_state_ids_for_task(ood_task_id)
        random.shuffle(ood_all_trials)
        selected_ood_trials = ood_all_trials[:20]
        for state_id in selected_ood_trials:
            test_metadata.append(
                {
                    "task_id": ood_task_id,
                    "state_id": state_id,
                    "data_source": "test_out_of_distribution",
                }
            )

    print(f"Generated {len(train_metadata)} training samples.")
    print(f"Generated {len(test_metadata)} testing samples.")
    print("-" * 20)

    train_ds_meta = Dataset.from_list(train_metadata)
    test_ds_meta = Dataset.from_list(test_metadata)

    def map_and_process(example, idx):
        task_id = example["task_id"]
        state_id = example["state_id"]
        data_source = example["data_source"]
        split = "train" if data_source == "train" else "test"
        task = task_suite.get_task(task_id)

        data = {
            "data_source": data_source,
            "prompt": task.language,
            "state_ids": state_id,
            "task_ids": task_id,
            "ability": "robot",
            "extra_info": {
                "split": split,
                "state_ids": state_id,
                "index": idx,
                "task": task,
                "task_ids": task_id,
                "task_suite_name": task_suite_name,
            },
        }
        return data

    print("Mapping and processing training dataset...")
    train_dataset = train_ds_meta.map(map_and_process, with_indices=True, num_proc=8)

    print("Mapping and processing test dataset...")
    test_dataset = test_ds_meta.map(map_and_process, with_indices=True, num_proc=8)

    if single_task_id is None:
        suite_save_dir = os.path.join(local_save_dir, task_suite_name)
    else:
        suite_save_dir = os.path.join(local_save_dir, f"{task_suite_name}_task{single_task_id}")

    os.makedirs(suite_save_dir, exist_ok=True)

    train_path = os.path.join(suite_save_dir, "train.parquet")
    test_path = os.path.join(suite_save_dir, "test.parquet")

    print(f"Saving training dataset to {train_path}")
    train_dataset.to_parquet(train_path)

    print(f"Saving test dataset to {test_path}")
    test_dataset.to_parquet(test_path)

    print("\nDataset generation complete!")

    print("\n--- Verification ---")
    print("Train dataset data sources:", train_dataset.unique("data_source"))
    print("Test dataset data sources:", test_dataset.unique("data_source"))
    print("Train dataset length:", len(train_dataset))
    print("Test dataset length:", len(test_dataset))
    print("Train task_ids:", train_dataset.unique("task_ids"))
    print("Test task_ids:", test_dataset.unique("task_ids"))


def resolve_task_suites(task_suite_name: str) -> list[str]:
    benchmark_dict = get_benchmark_dict()
    available_suites = sorted(
        [name for name in benchmark_dict.keys() if name.startswith("libero_") and name != "libero_100"]
    )

    requested = task_suite_name.strip().lower()
    if requested == "all":
        return available_suites

    suites = [suite.strip().lower() for suite in requested.split(",") if suite.strip()]
    invalid = [suite for suite in suites if suite not in available_suites]
    if invalid:
        raise ValueError(
            f"Unknown task_suite_name: {invalid}. Available suites: {available_suites}. "
            "You can also pass --task_suite_name all"
        )
    return suites


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_suite_name",
        default="libero_spatial",
        help="Task suite name. Support single suite (e.g. libero_spatial), multiple suites split by comma, or all.",
    )
    parser.add_argument(
        "--local_save_dir",
        default="~/data/libero_rl",
        help="The save directory for the preprocessed dataset.",
    )
    parser.add_argument(
        "--single_task_id",
        type=int,
        default=None,
        help="If set, only generate dataset for this single task_id within the suite.",
    )
    parser.add_argument(
        "--max_train_per_task",
        type=int,
        default=40,
        help="Maximum number of training samples kept for one task.",
    )
    parser.add_argument(
        "--max_test_per_task",
        type=int,
        default=10,
        help="Maximum number of ID test samples kept for one task.",
    )
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    task_suites = resolve_task_suites(args.task_suite_name)
    print(f"Will process task suites: {task_suites}")

    for task_suite_name in task_suites:
        build_dataset_for_suite(
            task_suite_name=task_suite_name,
            local_save_dir=local_save_dir,
            single_task_id=args.single_task_id,
            max_train_per_task=args.max_train_per_task,
            max_test_per_task=args.max_test_per_task,
        )