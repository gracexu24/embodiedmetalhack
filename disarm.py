"""Disarm the SO-101: turn torque OFF on every arm we can open, so it goes limp/safe.

Use this when NO other process is holding the arm's serial port -- e.g. the harness backend
was killed but the servos are still energised from a crash. Only one process can own the
port at a time, so if the harness backend is still running (holding the arm), disarm through
it instead:

    curl -X POST -H 'Content-Type: application/json' \\
        -d '{"command":"stop"}' http://localhost:8000/api/build/command

Run from the vendored so100-hackathon pixi env so so100_hackathon imports resolve:

    export REPO=$(pwd)
    cd third_party/so100-hackathon && pixi run python "$REPO/disarm.py"
"""

from __future__ import annotations

from so100_hackathon.feetech import FeetechBus, detect_arm_ports


def main() -> None:
    ports = detect_arm_ports()
    if not ports:
        print("no SO-100/101 arms found (no /dev/cu.usbmodem* ports).")
        return
    for port in ports:
        try:
            bus = FeetechBus(port)
            bus.set_torque(False)  # limp
            bus.close()
            print(f"disarmed {port} (torque OFF)")
        except Exception as exc:  # noqa: BLE001 -- report and continue to the next arm
            print(f"could NOT open {port}: {exc}")
            print("  -> another process is likely holding it (the harness backend?).")
            print("     disarm through the backend instead: POST {'command':'stop'} to /api/build/command")


if __name__ == "__main__":
    main()
