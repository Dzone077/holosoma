#!/usr/bin/env python3
"""Compare default joint positions between Isaac Lab and MuJoCo."""

# Isaac Lab defaults (from robot.py t1_29dof_waist_wrist)
isaac_lab_defaults = {
    "AAHead_yaw": 0.0,
    "Head_pitch": 0.0,
    "Left_Shoulder_Pitch": 0.2,
    "Left_Shoulder_Roll": -1.35,
    "Left_Elbow_Pitch": 0.0,
    "Left_Elbow_Yaw": -0.5,
    "Left_Wrist_Pitch": 0.0,
    "Left_Wrist_Yaw": 0.0,
    "Left_Hand_Roll": 0.0,
    "Right_Shoulder_Pitch": 0.2,
    "Right_Shoulder_Roll": 1.35,
    "Right_Elbow_Pitch": 0.0,
    "Right_Elbow_Yaw": 0.5,
    "Right_Wrist_Pitch": 0.0,
    "Right_Wrist_Yaw": 0.0,
    "Right_Hand_Roll": 0.0,
    "Waist": 0.0,
    "Left_Hip_Pitch": -0.2,
    "Left_Hip_Roll": 0.0,
    "Left_Hip_Yaw": 0.0,
    "Left_Knee_Pitch": 0.4,
    "Left_Ankle_Pitch": -0.25,
    "Left_Ankle_Roll": 0.0,
    "Right_Hip_Pitch": -0.2,
    "Right_Hip_Roll": 0.0,
    "Right_Hip_Yaw": 0.0,
    "Right_Knee_Pitch": 0.4,
    "Right_Ankle_Pitch": -0.25,
    "Right_Ankle_Roll": 0.0,
}

# MuJoCo defaults (from T1.yaml)
# Note: MuJoCo config uses shorthand like "Hip_Pitch" for both left and right
mujoco_defaults = {
    "Left_Shoulder_Pitch": 0.0,
    "Left_Shoulder_Roll": -1.25,
    "Left_Elbow_Pitch": 0.0,
    "Left_Elbow_Yaw": -0.5,
    "Right_Shoulder_Pitch": 0.0,
    "Right_Shoulder_Roll": 1.25,
    "Right_Elbow_Pitch": 0.0,
    "Right_Elbow_Yaw": 0.5,
    "Left_Hip_Pitch": -0.2,
    "Left_Hip_Roll": 0.0,
    "Left_Hip_Yaw": 0.0,
    "Left_Knee_Pitch": 0.4,
    "Left_Ankle_Pitch": -0.25,
    "Left_Ankle_Roll": 0.0,
    "Right_Hip_Pitch": -0.2,
    "Right_Hip_Roll": 0.0,
    "Right_Hip_Yaw": 0.0,
    "Right_Knee_Pitch": 0.4,
    "Right_Ankle_Pitch": -0.25,
    "Right_Ankle_Roll": 0.0,
}

# Policy-controlled DOFs (20 joints)
FASTTD3_POLICY_DOF_NAMES_20 = [
    "Left_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "Right_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
]

print("=== Comparison of Default Joint Positions ===")
print(f"{'Joint Name':30s} | {'Isaac Lab':>12s} | {'MuJoCo':>12s} | {'Diff':>12s} | {'Status'}")
print("-" * 85)

mismatches = []
for name in FASTTD3_POLICY_DOF_NAMES_20:
    isaac_val = isaac_lab_defaults.get(name, 0.0)
    mujoco_val = mujoco_defaults.get(name, 0.0)
    diff = isaac_val - mujoco_val
    status = "OK" if abs(diff) < 0.01 else "MISMATCH!"

    print(f"{name:30s} | {isaac_val:>12.4f} | {mujoco_val:>12.4f} | {diff:>12.4f} | {status}")

    if abs(diff) > 0.01:
        mismatches.append((name, isaac_val, mujoco_val))

print("\n=== Summary ===")
if mismatches:
    print(f"Found {len(mismatches)} mismatched joints:")
    for name, isaac_val, mujoco_val in mismatches:
        print(f"  - {name}: Isaac={isaac_val}, MuJoCo={mujoco_val}")
    print("\nThis is the root cause of policy transfer failure!")
    print("Since obs[11:31] = (dof_pos - default_dof_pos), different defaults")
    print("cause different observation values for the same physical joint position.")
else:
    print("All joints match!")
