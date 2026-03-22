import argparse
import os
import re
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from verl.experimental.vla.models.pi0_torch.modeling_pi0_torch import PI0ForActionPrediction

PI0_MAX_STATE_DIM = 32
DEFAULT_IMAGE_SIZE = 224


def parse_task_from_hdf5_path(hdf5_path: str) -> str:
    name = Path(hdf5_path).stem
    name = re.sub(r"_demo$", "", name)
    name = re.sub(r"^[A-Z]+_SCENE\d+_", "", name)
    return name.replace("_", " ")


def build_chunk(actions: np.ndarray, t: int, horizon: int) -> np.ndarray:
    end = min(t + horizon, len(actions))
    chunk = actions[t:end]
    if len(chunk) == 0:
        raise ValueError(f"Empty action chunk at t={t}, len(actions)={len(actions)}")
    if len(chunk) < horizon:
        pad = np.repeat(chunk[-1][None], horizon - len(chunk), axis=0)
        chunk = np.concatenate([chunk, pad], axis=0)
    return chunk.astype(np.float32)


def to_bchw_bfloat16(img_hwc: np.ndarray, image_size: int) -> torch.Tensor:
    x = torch.from_numpy(np.asarray(img_hwc)).permute(2, 0, 1).unsqueeze(0).float()
    # Match current offline preprocessing: flip H and W -> 180 degree rotation.
    x = torch.flip(x, dims=(2, 3))
    if tuple(x.shape[-2:]) != (image_size, image_size):
        x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return x.to(torch.bfloat16)


@torch.no_grad()
def predict_from_processed_input(model, tokenizer, images_dict, img_masks, task_list, state_tensor):
    device = state_tensor.device
    model._to(device)

    state_raw = state_tensor
    state = model.state_normalize_transform(state_raw)
    if model.no_state:
        state = torch.zeros_like(state)

    images, _ = model.image_transform.call_batch(images_dict)
    img_masks = [m.to(device=device) for m in img_masks]

    state_for_prompt = state[:, : state_raw.shape[-1]] if model.pi05_enabled else state
    lang_tokens, lang_masks = model.prompt_tokenizer_transform.call_batch(
        {"task": task_list, "observation.state": state_for_prompt}, tokenizer
    )
    lang_tokens = lang_tokens.to(device)
    lang_masks = lang_masks.to(device)

    if isinstance(images, (list, tuple)):
        devs = [img.device for img in images]
        print("[DEBUG] image devices:", devs)

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--tokenizer-path", required=True)
    ap.add_argument("--demo", default="demo_0")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--max-samples", type=int, default=50)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    ap.add_argument("--task", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    model = PI0ForActionPrediction.from_pretrained(args.model_path, trust_remote_code=True)
    model.eval()
    model._to(device)

    task_text = args.task or parse_task_from_hdf5_path(args.hdf5)
    print("[INFO] task text:", task_text)

    with h5py.File(args.hdf5, "r") as f:
        g = f["data"][args.demo]
        T = len(g["actions"])
        end = min(T - 1, args.start + args.max_samples)
        if end <= args.start:
            raise ValueError(f"Not enough timesteps: T={T}")

        obs_g = g["obs"]
        first_l1s, first_l2s, seq_l1s, seq_l2s = [], [], [], []

        for t in range(args.start, end):
            gt_chunk_np = build_chunk(np.asarray(g["actions"]), t, args.horizon)

            cam_high = to_bchw_bfloat16(np.asarray(obs_g["agentview_rgb"][t]), args.image_size).to(device)
            left_wrist = to_bchw_bfloat16(np.asarray(obs_g["eye_in_hand_rgb"][t]), args.image_size).to(device)
            right_wrist = torch.zeros_like(left_wrist, device=device, dtype=torch.bfloat16)

            ee_pos = torch.as_tensor(np.asarray(obs_g["ee_pos"][t]), device=device, dtype=torch.float32).unsqueeze(0)
            ee_ori = torch.as_tensor(np.asarray(obs_g["ee_ori"][t]), device=device, dtype=torch.float32).unsqueeze(0)
            gripper = torch.as_tensor(np.asarray(obs_g["gripper_states"][t]), device=device, dtype=torch.float32).unsqueeze(0)
            state = torch.cat([ee_pos, ee_ori, gripper], dim=-1)
            state = F.pad(state, (0, max(0, PI0_MAX_STATE_DIM - state.shape[-1])), "constant", 0)

            images_dict = {
                "observation.images.cam_high": cam_high,
                "observation.images.cam_left_wrist": left_wrist,
                "observation.images.cam_right_wrist": right_wrist,
            }
            img_masks = [
                torch.ones((1,), dtype=torch.bool, device=device),
                torch.ones((1,), dtype=torch.bool, device=device),
                torch.zeros((1,), dtype=torch.bool, device=device),
            ]
            task_list = [task_text]

            pred = predict_from_processed_input(model, tokenizer, images_dict, img_masks, task_list, state)
            pred = pred[:, : args.horizon, :7].float()
            gt = torch.as_tensor(gt_chunk_np, device=device, dtype=torch.float32).unsqueeze(0)

            diff = pred - gt
            first_l1 = diff[:, 0].abs().mean().item()
            first_l2 = diff[:, 0].pow(2).mean().sqrt().item()
            seq_l1 = diff.abs().mean().item()
            seq_l2 = diff.pow(2).mean().sqrt().item()
            first_l1s.append(first_l1)
            first_l2s.append(first_l2)
            seq_l1s.append(seq_l1)
            seq_l2s.append(seq_l2)

            if (t - args.start) < 5:
                print(f"t={t} first_l1={first_l1:.6f} first_l2={first_l2:.6f} seq_l1={seq_l1:.6f} seq_l2={seq_l2:.6f}")
                print("  gt_first:", gt[0, 0].detach().cpu().numpy())
                print("  pd_first:", pred[0, 0].detach().cpu().numpy())

        print("=" * 80)
        print(f"samples={len(first_l1s)}")
        print(f"mean first-step L1 : {np.mean(first_l1s):.6f}")
        print(f"mean first-step L2 : {np.mean(first_l2s):.6f}")
        print(f"mean horizon L1    : {np.mean(seq_l1s):.6f}")
        print(f"mean horizon L2    : {np.mean(seq_l2s):.6f}")
        print("=" * 80)


if __name__ == "__main__":
    main()
