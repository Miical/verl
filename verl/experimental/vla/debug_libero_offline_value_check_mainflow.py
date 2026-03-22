import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from transformers import AutoTokenizer

from verl.protocol import DataProto
from verl.experimental.vla.models.pi0_torch.datasets.libero_dataset import LiberoPi0DatasetInput
from verl.experimental.vla.models.pi0_torch.modeling_pi0_torch import PI0ForActionPrediction


def build_chunk(actions: np.ndarray, t: int, horizon: int) -> np.ndarray:
    end = min(t + horizon, len(actions))
    chunk = actions[t:end]
    if len(chunk) == 0:
        raise ValueError(f"Empty action chunk at t={t}, len(actions)={len(actions)}")
    if len(chunk) < horizon:
        pad = np.repeat(chunk[-1][None], horizon - len(chunk), axis=0)
        chunk = np.concatenate([chunk, pad], axis=0)
    return chunk.astype(np.float32)


def make_dataproto_from_hdf5(g, hdf5_path: str, t: int, horizon: int, device: torch.device) -> DataProto:
    T = len(g["actions"])
    t1 = min(t + 1, T - 1)
    actions = np.asarray(g["actions"])
    rewards = np.asarray(g["rewards"]) if "rewards" in g else None

    def ten(x, dtype=None):
        x = np.asarray(x)
        return torch.as_tensor(x, device=device, dtype=dtype)

    tensors = {
        "t0.obs.agentview_rgb": ten(g["obs"]["agentview_rgb"][t], torch.uint8).unsqueeze(0),
        "t0.obs.eye_in_hand_rgb": ten(g["obs"]["eye_in_hand_rgb"][t], torch.uint8).unsqueeze(0),
        "t0.obs.ee_pos": ten(g["obs"]["ee_pos"][t], torch.float32).unsqueeze(0),
        "t0.obs.ee_ori": ten(g["obs"]["ee_ori"][t], torch.float32).unsqueeze(0),
        "t0.obs.gripper_states": ten(g["obs"]["gripper_states"][t], torch.float32).unsqueeze(0),
        "t0.actions": ten(build_chunk(actions, t, horizon), torch.float32).unsqueeze(0),
        "t1.obs.agentview_rgb": ten(g["obs"]["agentview_rgb"][t1], torch.uint8).unsqueeze(0),
        "t1.obs.eye_in_hand_rgb": ten(g["obs"]["eye_in_hand_rgb"][t1], torch.uint8).unsqueeze(0),
        "t1.obs.ee_pos": ten(g["obs"]["ee_pos"][t1], torch.float32).unsqueeze(0),
        "t1.obs.ee_ori": ten(g["obs"]["ee_ori"][t1], torch.float32).unsqueeze(0),
        "t1.obs.gripper_states": ten(g["obs"]["gripper_states"][t1], torch.float32).unsqueeze(0),
        "t1.actions": ten(build_chunk(actions, t1, horizon), torch.float32).unsqueeze(0),
        "t1.chunk_dones": torch.tensor([t + horizon >= T - 1], device=device, dtype=torch.bool),
    }
    if rewards is not None:
        tensors["t1.rewards"] = torch.tensor([float(rewards[t1])], device=device, dtype=torch.float32)

    return DataProto.from_dict(
        tensors=tensors,
        non_tensors={
            "t0.hdf5_path": [hdf5_path],
            "t1.hdf5_path": [hdf5_path],
        },
    )


@torch.no_grad()
def predict_exact_offline_mainflow(model, tokenizer, dp: DataProto) -> torch.Tensor:
    ds_in = LiberoPi0DatasetInput.from_dataset_batch(dp)
    s0 = ds_in.s0
    state_raw = s0["state"]
    device = state_raw.device
    try:
        model._to(device)
    except Exception:
        pass

    state = model.state_normalize_transform(state_raw)
    if model.no_state:
        state = torch.zeros_like(state)

    images, _ = model.image_transform.call_batch(s0["images"])
    img_masks = [m.to(device=device) for m in s0["img_masks"]]
    state_for_prompt = state[:, : state_raw.shape[-1]] if model.pi05_enabled else state
    lang_tokens, lang_masks = model.prompt_tokenizer_transform.call_batch(
        {"task": s0["task"], "observation.state": state_for_prompt}, tokenizer
    )
    lang_tokens = lang_tokens.to(device)
    lang_masks = lang_masks.to(device)

    if model.flow_sde_enable:
        prefix_features = model.model.embed_prefix(
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
        )
        pred_action, _, _ = model._sample_actions_flow_sde(
            state_features=(prefix_features, state),
            noise_scale=model.flow_sde_rollout_noise_scale,
            requires_grad=False,
            return_log_prob=True,
        )
    else:
        pred_action = model.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state=state)

    actual_action = model.action_unnormalize_transform(pred_action)
    return actual_action


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--tokenizer-path", required=True)
    ap.add_argument("--demo", default="demo_0")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--max-samples", type=int, default=50)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    model = PI0ForActionPrediction.from_pretrained(args.model_path, trust_remote_code=True)
    model.eval()
    try:
        model._to(device)
    except Exception:
        model.to(device)

    with h5py.File(args.hdf5, "r") as f:
        g = f["data"][args.demo]
        T = len(g["actions"])
        end = min(T - 1, args.start + args.max_samples)
        if end <= args.start:
            raise ValueError(f"Not enough timesteps: T={T}")

        first_l1s, first_l2s = [], []
        seq_l1s, seq_l2s = [], []
        rows = []
        compare_horizon = None

        for t in range(args.start, end):
            dp = make_dataproto_from_hdf5(g, args.hdf5, t, args.horizon, device)
            pred = predict_exact_offline_mainflow(model, tokenizer, dp)
            gt = dp.batch["t0.actions"].float()

            pred = pred[:, :, : gt.shape[-1]].float()
            cur_h = min(pred.shape[1], gt.shape[1])
            if compare_horizon is None:
                compare_horizon = int(cur_h)
            pred = pred[:, :cur_h]
            gt = gt[:, :cur_h]

            diff = pred - gt
            first_l1 = diff[:, 0].abs().mean().item()
            first_l2 = diff[:, 0].pow(2).mean().sqrt().item()
            seq_l1 = diff.abs().mean().item()
            seq_l2 = diff.pow(2).mean().sqrt().item()
            first_l1s.append(first_l1)
            first_l2s.append(first_l2)
            seq_l1s.append(seq_l1)
            seq_l2s.append(seq_l2)

            row = {
                "t": int(t),
                "compare_horizon": int(cur_h),
                "first_l1": float(first_l1),
                "first_l2": float(first_l2),
                "seq_l1": float(seq_l1),
                "seq_l2": float(seq_l2),
                "gt_first": gt[0, 0].detach().cpu().tolist(),
                "pred_first": pred[0, 0].detach().cpu().tolist(),
            }
            rows.append(row)

            if (t - args.start) < 5:
                print(
                    f"t={t} H={cur_h} first_l1={first_l1:.6f} first_l2={first_l2:.6f} "
                    f"seq_l1={seq_l1:.6f} seq_l2={seq_l2:.6f}"
                )
                print("  gt_first:", np.array(row["gt_first"]))
                print("  pd_first:", np.array(row["pred_first"]))

        summary = {
            "samples": len(rows),
            "requested_horizon": int(args.horizon),
            "compare_horizon": int(compare_horizon or 0),
            "mean_first_l1": float(np.mean(first_l1s)) if first_l1s else None,
            "mean_first_l2": float(np.mean(first_l2s)) if first_l2s else None,
            "mean_seq_l1": float(np.mean(seq_l1s)) if seq_l1s else None,
            "mean_seq_l2": float(np.mean(seq_l2s)) if seq_l2s else None,
        }

        print("=" * 80)
        print(json.dumps(summary, indent=2))
        print("=" * 80)

        if args.output_json:
            os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
            with open(args.output_json, "w", encoding="utf-8") as f_out:
                json.dump({"summary": summary, "rows": rows}, f_out, indent=2)
            print(f"saved -> {args.output_json}")


if __name__ == "__main__":
    main()
