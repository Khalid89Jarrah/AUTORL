#!/usr/bin/env python3

import os
import gymnasium as gym
import numpy as np
import matplotlib

matplotlib.use("Agg")
import rclpy
from dataclasses import dataclass
from stable_baselines3.common.env_checker import check_env
from sensor_interaction.autorl_world import Auto_RL


def frd_to_flu(v):
    """Convert FRD → FLU coordinate frame."""
    v = np.asarray(v, dtype=np.float32)
    return np.array([v[0], -v[1], -v[2]], dtype=np.float32)


def run_rollout(base_env, sp_flu, steps=100):
    """Performs one rollout with constant action sequence."""
    obs, _ = base_env.reset(options={"eval_setpoint": sp_flu})
    obs_log, rew_log = [obs.copy()], []
    for step in range(steps):
        action = np.array([0.1, -0.1, 0.0], dtype=np.float32)
        obs, reward, terminated, truncated, _ = base_env.step(action)
        obs_log.append(obs.copy())
        rew_log.append(float(reward))
        if terminated or truncated:
            break
    return np.array(obs_log), np.array(rew_log)


def test_environment_determinism(env, eval_setpoints_frd, steps=100):
    """Tests environment determinism for multiple FRD setpoints."""
    base_env = env
    while hasattr(base_env, "env"):
        base_env = base_env.env

    print(f"[TEST] Unwrapped env type: {type(base_env)}")
    print(f"[TEST] Number of test setpoints: {len(eval_setpoints_frd)}\n")

    for idx, sp_frd in enumerate(eval_setpoints_frd, start=1):
        print(f"=== Setpoint {idx}: FRD={sp_frd.tolist()} ===")
        sp_flu = frd_to_flu(sp_frd)

        # Warmup reset
        obs_init, _ = base_env.reset(options={"eval_setpoint": sp_flu})
        print("[TEST] obs after reset:", obs_init)

        # Two deterministic rollouts
        s1, r1 = run_rollout(base_env, sp_flu, steps)
        s2, r2 = run_rollout(base_env, sp_flu, steps)

        # Align lengths
        n = min(len(s1), len(s2))
        m = min(len(r1), len(r2))
        s1, s2 = s1[:n], s2[:n]
        r1, r2 = r1[:m], r2[:m]

        # Compute absolute differences
        state_diff = np.abs(s1 - s2)
        reward_diff = np.abs(r1 - r2)
        mean_state_diff = state_diff.mean()
        max_state_diff = state_diff.max()
        mean_rew_diff = reward_diff.mean()
        max_rew_diff = reward_diff.max()

        # Binary equality checks
        same_states = np.allclose(s1, s2, atol=1e-8)
        same_rewards = np.allclose(r1, r2, atol=1e-8)

        print(f"--- Determinism Diagnostics ---")
        print(f"Steps compared : {n}, Rewards compared: {m}")
        print(f"Mean |Δstate|  : {mean_state_diff:.3e}")
        print(f"Max  |Δstate|  : {max_state_diff:.3e}")
        print(f"Mean |Δreward| : {mean_rew_diff:.3e}")
        print(f"Max  |Δreward| : {max_rew_diff:.3e}")
        print(f"States identical : {same_states}")
        print(f"Rewards identical: {same_rewards}")

        if not same_states or not same_rewards:
            print("Mismatch detected — non-deterministic behavior found!")
        print("-" * 60 + "\n")


def main(args=None):
    rclpy.init(args=args)
    env = gym.make("Autopilot-RF-v0")
    check_env(env)
    print("Starting deterministic environment test (headless)…")

    # Multiple FRD setpoints to test determinism at different states
    test_setpoints = [
        np.array([0.2, -0.1, 0.5], dtype=np.float32),
        np.array([-0.5, 0.3, -0.4], dtype=np.float32),
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.1, -0.2, 0.6], dtype=np.float32),
    ]

    test_environment_determinism(env, test_setpoints, steps=100)

    env.close()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
