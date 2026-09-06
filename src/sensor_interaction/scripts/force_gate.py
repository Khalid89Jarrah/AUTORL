#!/usr/bin/env python3
"""R2-4 gate: does the external force move the rate loop?

Runs a no-force PAIR first to measure the actual noise null, then each force
magnitude against it. Reports the ratio of force effect to that null.
"""
import os, sys
import numpy as np
import rclpy
import gymnasium as gym
from sensor_interaction.autorl_world import Auto_RL  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import disturbance_eval as de  # noqa: E402
from disturbance_eval import (  # noqa: E402
    apply_force, clear_force, find_wrench_topic, frd_to_flu, flu_to_frd, _take_flag,
)


def rollout(env, topic, force, n, sp_flu):
    obs, _ = env.reset(options={"eval_setpoint": sp_flu})
    clear_force(topic)
    if force is not None:
        apply_force(topic, force)
    a = np.zeros(3, dtype=np.float32)
    w = []
    for _ in range(n):
        obs, _, term, trunc, _ = env.step(a)
        w.append(flu_to_frd(obs[3:6]).copy())
        if term or trunc:
            obs, _ = env.reset(options={"eval_setpoint": sp_flu})
            clear_force(topic)
            if force is not None:
                apply_force(topic, force)
    clear_force(topic)
    return np.array(w)


def main():
    n = _take_flag("--steps", 200, int)
    oz = _take_flag("--offset-z", 0.10, float)
    flist = _take_flag("--forces", "2,5,10", str)
    forces = [float(x) for x in flist.split(",")]

    de.FORCE_OFFSET = (0.0, 0.0, oz)

    rclpy.init()
    env = gym.make("Autopilot-RL-v0").unwrapped
    topic = find_wrench_topic()
    print(f"[gate] topic  : {topic}")
    print(f"[gate] offset : (0, 0, {oz}) m in link frame")

    sp = frd_to_flu(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    try:
        a = rollout(env, topic, None, n, sp)
        b = rollout(env, topic, None, n, sp)
        m = min(len(a), len(b))
        null = np.abs(a[:m] - b[:m]).max()
        print(f"\n[gate] NULL max|dw| (no force, two runs): {null:.4e} rad/s")
        print(f"\n{'force N':>8} {'tau_est Nm':>11} {'max|dw|':>11} {'x null':>8}  verdict")
        for f in forces:
            c = rollout(env, topic, (f, 0.0, 0.0), n, sp)
            k = min(len(c), len(a))
            d = np.abs(c[:k] - a[:k]).max()
            r = d / null if null > 0 else float("inf")
            v = "PASS" if r > 10 else ("MARGINAL" if r > 3 else "INERT")
            print(f"{f:8.1f} {f*oz:11.4f} {d:11.4e} {r:8.1f}  {v}")
        print("\nPick the smallest force whose max|dw| is a visible fraction of")
        print("the 0.3-0.6 rad/s setpoints without pinning the actuators.")
    finally:
        if hasattr(env, "main_node"):
            env.main_node.destroy_node()
        env.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
