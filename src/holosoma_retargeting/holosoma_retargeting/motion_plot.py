import numpy as np
import matplotlib.pyplot as plt

ref = np.load(
    "/home/duarte/holosoma/src/holosoma_retargeting/holosoma_retargeting/"
    "converted_res/robot_only/fight1_subject2_mj_fps50.npz"
)

print("Keys in npz:", ref.files)          # list of all array names
for k in ref.files:
    arr = ref[k]
    print(f"{k}: shape={arr.shape}, dtype={arr.dtype}")

joint_names = ref["joint_names"]
print("\nJoint index → name:")
for i, name in enumerate(joint_names):
    print(f"{i:2d}: {name}")

print(f"joint_pos shape: {ref['joint_pos'].shape}")  # (12244, 30)
print(f"Expected actuated: 7 free + {len(joint_names)} = {7 + len(joint_names)}")
print("\nFirst timestep qpos:")
print("  Free base (0-6):", ref['joint_pos'][0, :7])
print("  Actuated (7-29):", ref['joint_pos'][0, 7:])

fps = ref["fps"].item()
qpos = ref["joint_pos"]    # (12244, 30)

dt = 1.0 / fps
time = np.arange(qpos.shape[0]) * dt

leg_indices = [11, 17]  # LHipPitch, LKneePitch, LAnklePitch, RHipPitch, RKneePitch, RAnklePitch
leg_names = joint_names[leg_indices].tolist()

plt.figure(figsize=(12, 8))
for i, idx in enumerate(leg_indices):
    plt.plot(time, qpos[:, 7 + idx], label=leg_names[i])  # 7+idx = actuated slice
plt.xlabel("Time (s)")
plt.ylabel("joint_pos (rad)")
plt.title("Leg joints: fight1_subject2_mj_fps50.npz")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()