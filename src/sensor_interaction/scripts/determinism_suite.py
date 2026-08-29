#!/usr/bin/env python3
"""
Determinism verification at scale.

Answers R2-13. The existing determinism_test.py runs ONE rollout pair per
setpoint, with a constant action [0.1, -0.1, 0.0], steps=100, on one machine
(run_rollout, lines 21-31). That is exactly what the reviewer objects to.

This script instead:
  * runs N rollout PAIRS (default 20) per setpoint,
  * drives them with a TRAINED POLICY, not a constant action,
  * runs full length (default 750 steps = the registered max_episode_steps),
  * optionally loads the CPU with background workers (--cpu-load),
  * compares with np.array_equal -- exact equality, not a tolerance,
  * writes every rollout to .npz so divergence_stats.py can build a
    distribution instead of quoting a single maximum.

Cross-machine coverage is obtained by running this same script on a second host
with --machine <name>; divergence_stats.py groups by that tag.

USAGE
    ros2 launch sensor_interaction node_launch.py \
        algorithm:=ppo gui:=false mode:=determinism_suite \
        model_path:=/opt/autorl_ws/models/ppo_model_ref_seed0.zip

    # options (passed through to the script):
    #   --pairs 20 --steps 750 --cpu-load 8 --machine hpc-login-p01
    #   --noise-tag on|off   (label only; noise is set in the SDF, not here)
"""

import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import rclpy
import gymnasium as gym

from stable_baselines3 import PPO
from sensor_interaction.autorl_world import Auto_RL  # noqa: F401  (registers env)


def _take_flag(flag, default, cast):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            value = cast(sys.argv[i + 1])
            del sys.argv[i : i + 2]
            return value
    return default


def frd_to_flu(v):
    v = np.asarray(v, dtype=np.float32)
    return np.array([v[0], -v[1], -v[2]], dtype=np.float32)


def _burn(stop_flag):
    x = 0.0
    while not stop_flag.value:
        x += np.sum(np.random.rand(4096) ** 2)


class CpuLoad:
    """Background CPU contention. R2-13 asks for determinism 'under CPU load'."""

    def __init__(self, workers):
        self.workers = workers
        self.procs = []
        self.stop_flag = None

    def __enter__(self):
        if self.workers > 0:
            self.stop_flag = mp.get_context("spawn").Value("b", False)
            for _ in range(self.workers):
                p = mp.get_context("spawn").Process(target=_burn, args=(self.stop_flag,), daemon=True)
                p.start()
                self.procs.append(p)
            print(f"[load] {self.workers} background CPU workers started")
        return self

    def __exit__(self, *exc):
        if self.stop_flag is not None:
            self.stop_flag.value = True
        for p in self.procs:
            p.join(timeout=2.0)
            if p.is_alive():
                p.terminate()
        if self.procs:
            print("[load] background CPU workers stopped")


def run_rollout(env, model, sp_flu, steps, seed):
    """One policy-driven rollout from a seeded reset."""
    obs, _ = env.reset(seed=seed, options={"eval_setpoint": sp_flu})
    print(f"[reset] simtime={env.main_node.get_sim_time()} obs0={np.array2string(obs[:9], precision=9)}", flush=True)
    obs_log = [obs.copy()]
    rew_log = []
    act_log = []

    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        obs_log.append(obs.copy())
        rew_log.append(float(reward))
        act_log.append(np.asarray(action, dtype=np.float32).copy())
        if terminated or truncated:
            break

    return (
        np.array(obs_log, dtype=np.float32),
        np.array(rew_log, dtype=np.float64),
        np.array(act_log, dtype=np.float32),
    )


def main(args=None):
    pairs = _take_flag("--pairs", 20, int)
    steps = _take_flag("--steps", 750, int)
    cpu_load = _take_flag("--cpu-load", 0, int)
    machine = _take_flag("--machine", os.uname()[1], str)
    noise_tag = _take_flag("--noise-tag", "unspecified", str)
    seed = _take_flag("--seed", 42, int)
    out_dir = _take_flag("--out", "./determinism_out", str)
    model_path_cli = _take_flag("--model", "", str)

    os.makedirs(out_dir, exist_ok=True)

    rclpy.init(args=args)
    env = gym.make("Autopilot-RL-v0")
    base = env.unwrapped

    model_path = model_path_cli
    if not model_path:
        from rcl_interfaces.msg import ParameterDescriptor

        tmp_node = rclpy.create_node("determinism_suite_param_loader")
        desc = ParameterDescriptor(dynamic_typing=True)
        model_path = tmp_node.declare_parameter("model_path", None, desc).value
        tmp_node.destroy_node()
    if not model_path:
        raise RuntimeError(
            "ERROR: pass model_path:=/path/to/model.zip or --model /path/to/model.zip"
        )

    model = PPO.load(model_path, device="cpu")
    print(f"[suite] model      : {model_path}")
    print(f"[suite] machine    : {machine}")
    print(f"[suite] pairs      : {pairs}   steps: {steps}   cpu-load: {cpu_load}")
    print(f"[suite] noise tag  : {noise_tag}")

    setpoints_frd = [
        np.array([-0.5, 0.3, -0.4], dtype=np.float32),
        np.array([0.1, -0.2, 0.6], dtype=np.float32),
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        np.array([-0.3, -0.1, 0.2], dtype=np.float32),
    ]

    for _ in range(4):  # warm-up rollouts, discarded: settle the sim clock
        run_rollout(base, model, frd_to_flu(setpoints_frd[0]), steps, seed)


    summary = []
    try:
        with CpuLoad(cpu_load):
            for sp_idx, sp_frd in enumerate(setpoints_frd, start=1):
                sp_flu = frd_to_flu(sp_frd)
                for pair in range(pairs):
                    s1, r1, a1 = run_rollout(base, model, sp_flu, steps, seed)
                    s2, r2, a2 = run_rollout(base, model, sp_flu, steps, seed)

                    n = min(len(s1), len(s2))
                    m = min(len(r1), len(r2))
                    s1c, s2c = s1[:n], s2[:n]
                    r1c, r2c = r1[:m], r2[:m]

                    states_equal = bool(np.array_equal(s1c, s2c))
                    rewards_equal = bool(np.array_equal(r1c, r2c))
                    d_state = np.abs(s1c - s2c)
                    d_rew = np.abs(r1c - r2c)

                    tag = f"sp{sp_idx}_pair{pair:03d}_{machine}_noise-{noise_tag}_load{cpu_load}"
                    np.savez(
                        os.path.join(out_dir, f"{tag}.npz"),
                        obs_a=s1c,
                        obs_b=s2c,
                        rew_a=r1c,
                        rew_b=r2c,
                        act_a=a1,
                        act_b=a2,
                        setpoint_frd=sp_frd,
                        machine=machine,
                        noise_tag=noise_tag,
                        cpu_load=cpu_load,
                    )

                    row = {
                        "tag": tag,
                        "setpoint_index": sp_idx,
                        "pair": pair,
                        "machine": machine,
                        "noise_tag": noise_tag,
                        "cpu_load": cpu_load,
                        "steps_compared": int(n),
                        "rewards_compared": int(m),
                        "states_identical": states_equal,
                        "rewards_identical": rewards_equal,
                        "mean_abs_dstate": float(d_state.mean()),
                        "max_abs_dstate": float(d_state.max()),
                        "mean_abs_dreward": float(d_rew.mean()) if m else 0.0,
                        "max_abs_dreward": float(d_rew.max()) if m else 0.0,
                    }
                    summary.append(row)
                    print(
                        f"[sp{sp_idx} pair {pair:03d}] n={n:4d} "
                        f"states_eq={states_equal} rewards_eq={rewards_equal} "
                        f"max|dstate|={row['max_abs_dstate']:.3e}"
                    )

        summary_path = os.path.join(
            out_dir, f"determinism_summary_{machine}_noise-{noise_tag}_load{cpu_load}.json"
        )
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        identical = sum(1 for r in summary if r["states_identical"])
        print(f"\n{identical}/{len(summary)} pairs bit-identical (np.array_equal)")
        print(f"Summary written to {summary_path}")

    finally:
        if hasattr(base, "main_node"):
            base.main_node.destroy_node()
        env.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
