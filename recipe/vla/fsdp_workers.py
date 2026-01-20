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
The main entry point to run the PPO algorithm
"""

import asyncio
import contextlib
import logging
import os
from packaging import version

import torch
import torch.distributed
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._unshard_param_utils import _get_module_fsdp_state, _unshard_params_for_summon
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType

from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_device_id, get_device_name, get_torch_device, set_expandable_segments
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fsdp_utils import fsdp_version, set_reshard_after_forward
from verl.utils.import_utils import import_external_libs
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.profiler import DistProfiler, log_gpu_memory_usage, simple_timer
from verl.utils.profiler.performance import reduce_timing, topk_reduce_ratio_min_max
from verl.workers.config import HFModelConfig
from verl.workers.fsdp_workers import ActorRolloutRefWorker

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


class RobActorRolloutRefWorker(ActorRolloutRefWorker):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    fsdp_unshard_exit_stack = contextlib.ExitStack()

    def _build_rollout(self, trust_remote_code=False):
        from recipe.vla.naive_rollout_rob import NaiveRolloutRob
        from recipe.vla.sac.naive_rollout_pi05 import PI0RolloutRob

        self.base_sync_done = False
        world_size = torch.distributed.get_world_size()
        dp = world_size
        infer_tp = self.config.rollout.tensor_model_parallel_size
        rollout_device_mesh = init_device_mesh(
            device_name, mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"]
        )
        # 3. init trainer and rollout random states
        self.torch_random_states = get_torch_device().get_rng_state()
        gen_dp_rank = rollout_device_mesh["dp"].get_local_rank()
        get_torch_device().manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
        self.gen_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.torch_random_states)

        fsdp_ver = fsdp_version(self.actor_module_fsdp)
        if torch.distributed.get_world_size() == 1 and fsdp_ver == 1:
            FSDP.set_state_dict_type(
                self.actor_module_fsdp,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig(),
            )
        elif fsdp_ver == 1:
            FSDP.set_state_dict_type(
                self.actor_module_fsdp,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )
        elif fsdp_ver == 2:
            # FSDP2 already handles state dict logic via torch.distributed.checkpoint APIs.
            pass
        else:
            raise NotImplementedError(f"Unsupported fsdp version {fsdp_ver}")

        self._register_dispatch_collect_info("rollout", dp_rank=self.rank, is_collect=True)
        # TODO(liujincheng): configurable
        self.rollout = PI0RolloutRob(module=self.actor_module_fsdp, model_config=self.config.model, tokenizer=self.tokenizer)

        model_config: HFModelConfig = omega_conf_to_dataclass(self.config.model, dataclass_type=HFModelConfig)
        self.model_config = model_config

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def switch_to_rollout(self):
        print(f"[switch_to_rollout] Starting...", flush=True)
        try:
            loop = asyncio.get_event_loop()
            print(f"[switch_to_rollout] Got event loop, calling rollout_mode()...", flush=True)
            loop.run_until_complete(self.rollout_mode())
            print(f"[switch_to_rollout] rollout_mode() completed, calling log_gpu_memory_usage...", flush=True)
            log_gpu_memory_usage("After switch to rollout mode", logger=logger)
            print(f"[switch_to_rollout] Completed successfully!", flush=True)
        except Exception as e:
            print(f"[switch_to_rollout] ERROR: {type(e).__name__}: {e}", flush=True)
            import traceback
            print(f"[switch_to_rollout] Traceback:\n{traceback.format_exc()}", flush=True)
            raise

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def switch_to_train(self):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.trainer_mode())
        log_gpu_memory_usage("After switch to trainer mode", logger=logger)

    async def rollout_mode(self):
        """Context switch hybridengine to rollout mode."""
        import sys
        print(f"[rollout_mode] Starting rollout_mode...", flush=True)
        sys.stdout.flush()

        try:
            print(f"[rollout_mode] Setting model to eval mode...", flush=True)
            self.actor_module_fsdp.eval()

            print(f"[rollout_mode] Calling aggressive_empty_cache...", flush=True)
            aggressive_empty_cache(force_sync=True)
            self.base_sync_done = True

            # important: need to manually set the random states of each tp to be identical.
            print(f"[rollout_mode] Setting random states...", flush=True)
            self.torch_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.gen_random_states)

            fsdp_ver = fsdp_version(self.actor_module_fsdp)
            print(f"[rollout_mode] FSDP version detected: {fsdp_ver}, model type: {type(self.actor_module_fsdp)}", flush=True)
            
            if fsdp_ver == 1:
                # Check if FSDP is using NO_SHARD strategy (world_size=1)
                # In this case, params are already unsharded, no need to call summon_full_params
                from torch.distributed.fsdp import ShardingStrategy
                fsdp_state = self.actor_module_fsdp
                sharding_strategy = getattr(fsdp_state, 'sharding_strategy', None)
                print(f"[rollout_mode] FSDP1: sharding_strategy={sharding_strategy}", flush=True)
                
                if sharding_strategy == ShardingStrategy.NO_SHARD:
                    # NO_SHARD means params are already full, no unshard needed
                    print(f"[rollout_mode] FSDP1: NO_SHARD detected, skipping summon_full_params", flush=True)
                    self.fsdp_unshard_exit_stack = None
                else:
                    # For FSDP1 with actual sharding: Use FSDP.summon_full_params context manager
                    print(f"[rollout_mode] FSDP1: Calling summon_full_params...", flush=True)
                    try:
                        fsdp_unshard_exit_stack = contextlib.ExitStack()
                        # Use the public API summon_full_params instead of internal _unshard_params_for_summon
                        fsdp_unshard_exit_stack.enter_context(
                            FSDP.summon_full_params(
                                self.actor_module_fsdp,
                                writeback=False,
                                rank0_only=False,
                                offload_to_cpu=False,
                            )
                        )
                        self.fsdp_unshard_exit_stack = fsdp_unshard_exit_stack
                        print(f"[rollout_mode] FSDP1: summon_full_params succeeded", flush=True)
                    except Exception as e:
                        print(f"[rollout_mode] FSDP1: summon_full_params failed: {e}", flush=True)
                        logger.warning(f"Failed to summon full params for FSDP1: {e}. Continuing without unshard.")
                        self.fsdp_unshard_exit_stack = None
            elif fsdp_ver == 2:
                print(f"[rollout_mode] FSDP2: Calling unshard...", flush=True)
                try:
                    self.actor_module_fsdp.unshard()
                    for m in self.actor_module_fsdp.modules():
                        if isinstance(m, FSDPModule) or hasattr(m, "unshard"):
                            m.unshard()
                    if version.parse(torch.__version__) < version.parse("2.8"):
                        set_reshard_after_forward(self.actor_module_fsdp, False)
                    else:
                        self.actor_module_fsdp.set_reshard_after_forward(False)
                    print(f"[rollout_mode] FSDP2: unshard succeeded", flush=True)
                except Exception as e:
                    print(f"[rollout_mode] FSDP2: unshard failed: {e}", flush=True)
                    logger.warning(f"Failed to unshard for FSDP2: {e}. Continuing without unshard.")
            elif fsdp_ver == 0:
                # Model is not wrapped with FSDP, no unshard needed
                print(f"[rollout_mode] No FSDP wrapping detected, skipping unshard", flush=True)
                self.fsdp_unshard_exit_stack = None
            else:
                raise NotImplementedError(f"Unsupported fsdp version {fsdp_ver}")

            print(f"[rollout_mode] Completed successfully!", flush=True)
            logger.info("rollout mode")
            
        except Exception as e:
            print(f"[rollout_mode] FATAL ERROR: {type(e).__name__}: {e}", flush=True)
            import traceback
            print(f"[rollout_mode] Traceback:\n{traceback.format_exc()}", flush=True)
            raise

    async def trainer_mode(self):
        """Context switch hybridengine to trainer mode."""
        import sys
        print(f"[trainer_mode] Starting trainer_mode...", flush=True)

        try:
            self.actor_module_fsdp.train()

            # add empty cache after each compute
            aggressive_empty_cache(force_sync=True)
            set_expandable_segments(True)

            # restore random states
            self.gen_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.torch_random_states)

            fsdp_ver = fsdp_version(self.actor_module_fsdp)
            print(f"[trainer_mode] FSDP version: {fsdp_ver}", flush=True)
            
            if fsdp_ver == 1:
                if self.fsdp_unshard_exit_stack is not None:
                    try:
                        self.fsdp_unshard_exit_stack.close()
                        print(f"[trainer_mode] FSDP1: exit_stack closed", flush=True)
                    except Exception as e:
                        print(f"[trainer_mode] FSDP1: failed to close exit_stack: {e}", flush=True)
                        logger.warning(f"Failed to close FSDP1 unshard exit stack: {e}")
                        # The exit_stack close failed due to FSDP internal state issues.
                        # This is a known issue with FSDP summon_full_params.
                        # We'll try to recover by:
                        # 1. Synchronizing all ranks
                        # 2. Clearing the exit stack without closing
                        # 3. Hoping the model state is still usable
                        print(f"[trainer_mode] FSDP1: Attempting recovery...", flush=True)
                        try:
                            if torch.distributed.is_initialized():
                                torch.distributed.barrier()
                                print(f"[trainer_mode] FSDP1: barrier completed", flush=True)
                            # Clear CUDA cache to free any inconsistent memory
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                print(f"[trainer_mode] FSDP1: CUDA cache cleared", flush=True)
                        except Exception as recovery_e:
                            print(f"[trainer_mode] FSDP1: recovery failed: {recovery_e}", flush=True)
                        print(f"[trainer_mode] FSDP1: Continuing despite error (model may be in inconsistent state)", flush=True)
                    self.fsdp_unshard_exit_stack = None
            elif fsdp_ver == 2:
                try:
                    self.actor_module_fsdp.reshard()
                    for m in self.actor_module_fsdp.modules():
                        if isinstance(m, FSDPModule) or hasattr(m, "reshard"):
                            m.reshard()
                    if version.parse(torch.__version__) < version.parse("2.8"):
                        set_reshard_after_forward(self.actor_module_fsdp, True)
                    else:
                        self.actor_module_fsdp.set_reshard_after_forward(True)
                    print(f"[trainer_mode] FSDP2: reshard succeeded", flush=True)
                except Exception as e:
                    print(f"[trainer_mode] FSDP2: reshard failed: {e}", flush=True)
                    logger.warning(f"Failed to reshard for FSDP2: {e}")
            elif fsdp_ver == 0:
                # Model is not wrapped with FSDP, no reshard needed
                print(f"[trainer_mode] No FSDP wrapping, skipping reshard", flush=True)
            else:
                raise NotImplementedError(f"Unsupported fsdp version {fsdp_ver}")

            print(f"[trainer_mode] Completed successfully!", flush=True)
            logger.info("trainer mode")
            
        except Exception as e:
            print(f"[trainer_mode] FATAL ERROR: {type(e).__name__}: {e}", flush=True)
            import traceback
            print(f"[trainer_mode] Traceback:\n{traceback.format_exc()}", flush=True)
            raise

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"), blocking=False)
    @DistProfiler.annotate(color="red", role="rollout_generate")
    def generate_sequences(self, prompts: DataProto):
        # Support all hardwares
        assert self._is_rollout
        prompts = prompts.to(get_device_id())

        meta_info = {
            "eos_token_id": self.model_config.generation_config.eos_token_id
            if self.model_config.generation_config is not None
            else self.model_config.tokenizer.eos_token_id,
            "pad_token_id": self.model_config.generation_config.pad_token_id
            if self.model_config.generation_config is not None
            else self.model_config.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)

        timing_generate = {}

        with simple_timer("generate_sequences", timing_generate):
            output = self.rollout.generate_sequences(prompts=prompts)

        timing_generate_topk_ratio, timing_generate_min, timing_generate_max = topk_reduce_ratio_min_max(
            timing_generate["generate_sequences"]
        )
        timing_generate = reduce_timing(timing_generate)
        timing_generate.update(
            {
                "generation_timing/max": timing_generate_max,
                "generation_timing/min": timing_generate_min,
                "generation_timing/topk_ratio": timing_generate_topk_ratio,
            }
        )
        output.meta_info["metrics"] = timing_generate
        output = output.to("cpu")

        # clear kv cache
        get_torch_device().empty_cache()
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # from recipe.vla.dp_rob import RobDataParallelPPOActor
        from recipe.vla.sac.sac_actor import PI0RobDataParallelPPOActor

        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        from omegaconf import OmegaConf
        override_model_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))

        from recipe.vla.models import register_vla_models
        register_vla_models()

        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(self.config.model.path, trust_remote_code=True)

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = self.config.actor.fsdp_config
            else:
                optim_config = None
                fsdp_config = OmegaConf.create()
            self.actor_module_fsdp, self.actor_optimizer, self.actor_lr_scheduler, self.actor_model_config = (
                self._build_model_optimizer(
                    model_path=self.config.model.path,
                    fsdp_config=fsdp_config,
                    optim_config=optim_config,
                    override_model_config=override_model_config,
                    enable_gradient_checkpointing=self.config.model.get("enable_gradient_checkpointing", False),
                    trust_remote_code=self.config.model.get("trust_remote_code", False),
                )
            )

            if fsdp_version(self.actor_module_fsdp) == 1:
                # get the original unwrapped module
                self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            # TODO(liujincheng): configurable
            self.actor = PI0RobDataParallelPPOActor(
                config=self.config.actor, actor_module=self.actor_module_fsdp, actor_optimizer=self.actor_optimizer
            )

        if self._is_rollout:
            self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))

        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_config=self.config.actor.checkpoint,
            )

        torch.distributed.barrier()
