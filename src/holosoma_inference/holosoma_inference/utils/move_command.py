"""Utility to send move commands to a running LocomotionPolicy via UDP."""

import json
import socket


def move(vx: float, vy: float, vyaw: float, host: str = "127.0.0.1", port: int = 7654):
    """Send a single move command to the running policy.

    Args:
        vx: Linear velocity in x-direction (m/s).
        vy: Linear velocity in y-direction (m/s).
        vyaw: Angular velocity around z-axis (rad/s).
        host: Host where run_policy.py is running.
        port: UDP port (must match the one passed to start_command_server, default 7654).
    """
    payload = json.dumps({"vx": vx, "vy": vy, "vyaw": vyaw}).encode()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.sendto(payload, (host, port))
