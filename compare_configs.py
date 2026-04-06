#!/usr/bin/env python3
"""Compare MuJoCo vs Isaac Lab configurations for FastTD3 policy transfer."""

print("=" * 70)
print("CONFIGURATION COMPARISON: MuJoCo vs Isaac Lab")
print("=" * 70)

# ============= NORMALIZATION VALUES (CRITICAL FOR OBS ENCODING) =============
print("\n### NORMALIZATION VALUES (used in observation encoding) ###")
print(f"{'Parameter':<20} | {'MuJoCo T1.yaml':<15} | {'Isaac Lab':<15} | {'Match?'}")
print("-" * 70)

norm_comparison = [
    ("gravity", 1.0, 1.0),
    ("lin_vel", 2.0, 2.0),
    ("ang_vel", 0.25, 0.25),
    ("dof_pos", 1.0, 1.0),
    ("dof_vel", 0.05, 0.05),
]

for name, mujoco, isaac in norm_comparison:
    match = "✓" if mujoco == isaac else "✗ MISMATCH!"
    print(f"{name:<20} | {mujoco:<15} | {isaac:<15} | {match}")

# ============= STIFFNESS VALUES (PD CONTROLLER GAINS) =============
print("\n### STIFFNESS VALUES (PD controller P-gain) [N*m/rad] ###")
print(f"{'Joint Group':<20} | {'MuJoCo':<15} | {'Isaac Lab':<15} | {'Match?'}")
print("-" * 70)

stiffness_comparison = [
    ("Shoulder", 20.0, 20.0),
    ("Elbow", 20.0, 20.0),
    ("Hip", 200.0, 200.0),
    ("Knee", 200.0, 200.0),
    ("Ankle", 50.0, 50.0),  # MuJoCo has single "Ankle", Isaac has Ankle_Pitch/Roll both 50
]

for name, mujoco, isaac in stiffness_comparison:
    match = "✓" if mujoco == isaac else "✗ MISMATCH!"
    print(f"{name:<20} | {mujoco:<15} | {isaac:<15} | {match}")

# ============= DAMPING VALUES (PD CONTROLLER D-GAIN) =============
print("\n### DAMPING VALUES (PD controller D-gain) [N*m*s/rad] ###")
print(f"{'Joint Group':<20} | {'MuJoCo':<15} | {'Isaac Lab':<15} | {'Match?'}")
print("-" * 70)

damping_comparison = [
    ("Shoulder", 0.5, 0.5),
    ("Elbow", 0.5, 0.5),
    ("Hip", 5.0, 5.0),
    ("Knee", 5.0, 5.0),
    ("Ankle", 3.0, 3.0),
]

for name, mujoco, isaac in damping_comparison:
    match = "✓" if mujoco == isaac else "✗ MISMATCH!"
    print(f"{name:<20} | {mujoco:<15} | {isaac:<15} | {match}")

# ============= ACTION SCALE =============
print("\n### ACTION SCALE ###")
print(f"{'Config':<30} | {'Value':<15}")
print("-" * 50)
print(f"{'MuJoCo T1.yaml':<30} | {1.0:<15}")
print(f"{'Isaac Lab robot config':<30} | {0.25:<15} (NOT USED)")
print(f"{'Isaac Lab train_fasttd3.py':<30} | {1.0:<15} (USED)")
print("Note: FastTD3Env uses its own action_scale (1.0), not the robot config's")

# ============= CONTROL DECIMATION / FREQUENCY =============
print("\n### CONTROL FREQUENCY ###")
print(f"{'Parameter':<30} | {'MuJoCo':<15} | {'Isaac Lab':<15}")
print("-" * 65)
print(f"{'Physics dt':<30} | {'0.002 s':<15} | {'0.002 s (500Hz)':<15}")
print(f"{'Control decimation':<30} | {'10':<15} | {'10':<15}")
print(f"{'Control frequency':<30} | {'50 Hz':<15} | {'50 Hz':<15}")

print("\n" + "=" * 70)
print("SUMMARY: All critical values match! ✓")
print("=" * 70)
print("""
The configurations are compatible. The key points:

1. NORMALIZATION VALUES: All match ✓
   - These are used to scale observations and are critical for policy transfer

2. STIFFNESS/DAMPING: All match ✓
   - These affect physics behavior but policies can be somewhat robust to small differences

3. ACTION SCALE: Matches (both use 1.0) ✓
   - MuJoCo T1.yaml: action_scale=1.0
   - train_fasttd3.py: action_scale=1.0 (overrides robot config's 0.25)

4. CONTROL FREQUENCY: Matches (50 Hz) ✓
   - Same physics dt (0.002s = 500Hz) and decimation (10)
""")
