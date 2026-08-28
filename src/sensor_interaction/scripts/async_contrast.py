#!/usr/bin/env python3
"""
Async vs blocking contrast experiment.

Answers R1-3 / R2-1 / R2-13: the paper enumerates four failure modes of
asynchronous ROS2-Gazebo coupling (lines 372-387) but never demonstrates that
they occur. This script runs the same low-level control loop twice -- once with
the blocking service handshake AutoRL uses, once with the world free-running --
and counts how often each control action fails to correspond to exactly one
physics update.

GROUND TRUTH
------------
Sim-time advance per control action is measured from the *IMU message header
stamp*, not from /SimTime. Reason: the world is paused between steps in blocking
mode, so the clock topic cannot be relied on to publish a fresh sample there.
The IMU stamp is written by the Gazebo IMU system and is therefore produced by
the physics update itself.

  ticks = round((stamp_k - stamp_{k-1}) / dt)

    ticks == 1  -> nominal: one action, one physics update
    ticks == 0  -> no fresh measurement for this action
    ticks >= 2  -> more than one physics update per action
    ticks <  0  -> out-of-order delivery

WHAT IS AND IS NOT MEASURED
---------------------------
The paper's failure modes 1 (stale observation) and 3 (skipped step) both
present as ticks == 0 here and are reported as one combined bucket. Separating
them requires the world clock, which is not published while paused. This is
stated rather than guessed.

Failure mode 4 (dropped motor command) is NOT measured. Detecting it requires
Gazebo to echo the rotor speeds it actually applied; no such topic is bridged in
node_launch.py. The report prints "not measured" for it rather than a fabricated
zero.

USAGE
    ros2 launch sensor_interaction node_launch.py \
        algorithm:=ppo gui:=false mode:=async_contrast

    # or directly:
    ros2 run sensor_interaction async_contrast.py --ros-args -p steps:=10000

The --dt value must match <max_step_size> in
resources/autorl_drone/autorl_drone.sdf (currently 0.004).
"""

import json
import os
import sys
import time

import numpy as np
import rclpy
import gymnasium as gym

from ros_gz_interfaces.srv import ControlWorld
from ros_gz_interfaces.msg import WorldReset, WorldControl

from sensor_interaction.autorl_world import Auto_RL  # noqa: F401  (registers env)


def _take_flag(flag, default, cast):
    """Read `--flag VALUE` and delete it from sys.argv so rclpy never sees it."""
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            value = cast(sys.argv[i + 1])
            del sys.argv[i : i + 2]
            return value
    return default


def stamp_seconds(msg):
    """Sim time carried by a bridged sensor_msgs/Imu header."""
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def set_world_running(main_node, running):
    """Unpause (running=True) or pause (running=False) the world.

    Uses the same /world/default/control service the step handler uses, but with
    multi_step = 0 so no step is requested -- only the pause flag changes.
    """
    handler = main_node.sim_step_handler
    request = ControlWorld.Request()

    world_reset = WorldReset()
    world_reset.all = False
    world_reset.time_only = False
    world_reset.model_only = False

    world_control = WorldControl()
    world_control.reset = world_reset
    world_control.pause = not running
    world_control.multi_step = 0
    request.world_control = world_control

    future = handler.client.call_async(request)
    deadline = time.time() + 10.0
    while not future.done() and time.time() < deadline:
        time.sleep(0.01)
    return future.done()


def hover_rotor_speeds(env):
    """Rotor speeds for a zero-torque hover command, via the normal mixer path."""
    command = np.concatenate(
        (np.zeros(3, dtype=np.float64), np.array([0.0, 0.0, env.Hover_thrust]))
    )
    return env.main_node.compute_sequential_desaturation(
        command, env.inverse_effectiveness_matrix
    )


def classify(deltas, dt):
    """Bucket per-action sim-time advances into failure-mode counts."""
    ticks = np.rint(np.asarray(deltas) / dt).astype(int)
    counts = {
        "nominal_one_step": int(np.sum(ticks == 1)),
        "zero_advance_stale_or_skipped": int(np.sum(ticks == 0)),
        "multi_step": int(np.sum(ticks >= 2)),
        "out_of_order": int(np.sum(ticks < 0)),
        "dropped_motor_command": "not measured (see module docstring)",
    }
    multi = ticks[ticks >= 2]
    counts["multi_step_histogram"] = {
        int(k): int(v) for k, v in zip(*np.unique(multi, return_counts=True))
    }
    counts["max_ticks_observed"] = int(ticks.max()) if len(ticks) else 0
    return counts, ticks


def run_blocking(env, steps, dt):
    """AutoRL's own loop: publish, blocking step, blocking IMU read."""
    main_node = env.main_node
    set_world_running(main_node, False)
    rotors = hover_rotor_speeds(env)

    stamps = []
    for _ in range(steps):
        main_node.publish_motor_commands(rotors)
        main_node.perform_simulation_step(1)
        imu = main_node.read_imu()
        stamps.append(stamp_seconds(imu))

    return np.diff(np.array(stamps))


def run_async(env, steps, dt, rate_hz):
    """Free-running world: no step handshake, no wait for a fresh IMU sample.

    The world is unpaused once, then the control loop publishes at its own
    wall-clock rate and reads whatever IMU message is currently latched --
    exactly the asynchronous pattern the paper argues against.
    """
    main_node = env.main_node
    rotors = hover_rotor_speeds(env)

    set_world_running(main_node, True)
    time.sleep(0.5)  # let the world start advancing before sampling

    period = 1.0 / rate_hz if rate_hz > 0 else 0.0
    stamps = []
    next_deadline = time.time()
    for _ in range(steps):
        main_node.publish_motor_commands(rotors)

        # Latched read: no wait, no event clear. This is the whole point.
        imu = main_node.imu_handler.imu_data
        if imu is None:
            time.sleep(period)
            continue
        stamps.append(stamp_seconds(imu))

        next_deadline += period
        sleep_for = next_deadline - time.time()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            next_deadline = time.time()

    set_world_running(main_node, False)
    return np.diff(np.array(stamps))


def report(name, deltas, dt, out_dir):
    if len(deltas) == 0:
        print(f"[{name}] no samples collected")
        return None

    counts, ticks = classify(deltas, dt)
    total = len(ticks)
    counts["total_actions"] = total
    counts["mode"] = name
    counts["dt_s"] = dt

    print(f"\n=== {name} ===")
    print(f"actions compared              : {total}")
    print(f"exactly one physics update    : {counts['nominal_one_step']}")
    print(f"no fresh IMU (stale/skipped)  : {counts['zero_advance_stale_or_skipped']}")
    print(f"two or more physics updates   : {counts['multi_step']}")
    print(f"out-of-order                  : {counts['out_of_order']}")
    print(f"dropped motor command         : {counts['dropped_motor_command']}")
    if counts["multi_step_histogram"]:
        print(f"multi-step histogram          : {counts['multi_step_histogram']}")

    np.savez(
        os.path.join(out_dir, f"async_contrast_{name}.npz"),
        stamp_deltas=deltas,
        ticks=ticks,
        dt=dt,
    )
    return counts


def main(args=None):
    steps = _take_flag("--steps", 10000, int)
    dt = _take_flag("--dt", 0.004, float)
    rate_hz = _take_flag("--rate", 250.0, float)
    which = _take_flag("--mode", "both", str)
    out_dir = _take_flag("--out", "./async_contrast_out", str)

    os.makedirs(out_dir, exist_ok=True)

    rclpy.init(args=args)
    env = gym.make("Autopilot-RL-v0")
    base = env.unwrapped

    results = []
    try:
        base.reset()
        first_imu = base.main_node.read_imu()
        print(
            "[check] first IMU header stamp = "
            f"{stamp_seconds(first_imu):.6f} s  (must be SIM time, not wall time)"
        )

        if which in ("both", "blocking"):
            d = run_blocking(base, steps, dt)
            r = report("blocking", d, dt, out_dir)
            if r:
                results.append(r)
            base.reset()

        if which in ("both", "async"):
            d = run_async(base, steps, dt, rate_hz)
            r = report("async", d, dt, out_dir)
            if r:
                results.append(r)

        summary_path = os.path.join(out_dir, "async_contrast_summary.json")
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSummary written to {summary_path}")

    finally:
        try:
            set_world_running(base.main_node, False)
        except Exception:
            pass
        if hasattr(base, "main_node"):
            base.main_node.destroy_node()
        env.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
