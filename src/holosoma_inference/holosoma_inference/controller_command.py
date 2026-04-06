#!/usr/bin/env python3
"""Interactive terminal controller for LocomotionPolicy.

Run this while run_policy.py is active in another terminal.
Type velocities when prompted, or use shortcut commands.
"""

from holosoma_inference.utils.move_command import move


def print_help():
    print("\nCommands:")
    print("  <vx> <vy> <vyaw>  — set velocities (e.g. '0.5 0 0')")
    print("  stop / z          — set all velocities to zero")
    print("  help / ?          — show this message")
    print("  quit / q          — exit controller\n")


def main():
    print("=== Locomotion Controller ===")
    print_help()

    vx, vy, vyaw = 0.0, 0.0, 0.0

    while True:
        try:
            raw = input(f"[vx={vx:+.2f} vy={vy:+.2f} vyaw={vyaw:+.2f}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not raw:
            continue

        if raw in ("quit", "q"):
            move(0.0, 0.0, 0.0)
            print("Stopped robot. Exiting.")
            break
        elif raw in ("stop", "z"):
            vx, vy, vyaw = 0.0, 0.0, 0.0
            move(vx, vy, vyaw)
            print("Velocities zeroed.")
        elif raw in ("help", "?"):
            print_help()
        else:
            parts = raw.split()
            if len(parts) != 3:
                print("Expected 3 values: vx vy vyaw")
                continue
            try:
                vx, vy, vyaw = float(parts[0]), float(parts[1]), float(parts[2])
            except ValueError:
                print("Invalid numbers. Example: 0.5 0 0.1")
                continue
            move(vx, vy, vyaw)
            print(f"Sent: vx={vx:+.2f} m/s  vy={vy:+.2f} m/s  vyaw={vyaw:+.2f} rad/s")


if __name__ == "__main__":
    main()