#!/usr/bin/env python3
"""Quick script to check checkpoint structure."""
import torch

ckpt_path = "models/t1_fasttd3_t1_locomotion_1_100000.pt"
print(f"Loading checkpoint: {ckpt_path}")

ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

print("\n=== Checkpoint Keys ===")
for k in ckpt.keys():
    print(f"  {k}")
# testing
print("\n=== Actor State Dict First Keys ===")
for i, (k, v) in enumerate(ckpt["actor_state_dict"].items()):
    if i < 10:
        print(f"  {k}: shape={v.shape}")

print("\n=== Obs Normalizer State ===")
obs_state = ckpt.get("obs_normalizer_state", {})
if obs_state:
    for k, v in obs_state.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}")
            if v.numel() < 20:
                print(f"    values: {v.flatten()[:10]}")
            else:
                print(f"    mean={v.mean().item():.6f}, std={v.std().item():.6f}")
else:
    print("  Empty or None")

print("\n=== Args (selected) ===")
args = ckpt.get("args", {})
for k in ["num_envs", "actor_hidden_dim", "init_scale", "obs_normalization"]:
    print(f"  {k}: {args.get(k)}")

print("\n=== Observation Normalizer Mean (first 15) ===")
if "_mean" in obs_state:
    mean = obs_state["_mean"].flatten()
    print(f"  obs[0:3]  (proj_grav):  {mean[0:3].tolist()}")
    print(f"  obs[3:6]  (ang_vel):    {mean[3:6].tolist()}")
    print(f"  obs[6:9]  (commands):   {mean[6:9].tolist()}")
    print(f"  obs[9:11] (gait):       {mean[9:11].tolist()}")
    print(f"  obs[11:14] (dof_pos):   {mean[11:14].tolist()}")

print("\n=== Observation Normalizer Std (first 15) ===")
if "_std" in obs_state:
    std = obs_state["_std"].flatten()
    print(f"  obs[0:3]  (proj_grav):  {std[0:3].tolist()}")
    print(f"  obs[3:6]  (ang_vel):    {std[3:6].tolist()}")
    print(f"  obs[6:9]  (commands):   {std[6:9].tolist()}")
    print(f"  obs[9:11] (gait):       {std[9:11].tolist()}")
    print(f"  obs[11:14] (dof_pos):   {std[11:14].tolist()}")
