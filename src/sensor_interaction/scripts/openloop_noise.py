#!/usr/bin/env python3
"""Open-loop noise measurement. Fixed action sequence, replayed twice, noise on.
No policy -> no feedback -> the physics is identical in both runs, so every
observation difference is the injected sensor noise itself."""
import sys, numpy as np, rclpy, gymnasium as gym
from sensor_interaction.autorl_world import Auto_RL  # noqa: F401

def flag(f, d, c):
    if f in sys.argv:
        i = sys.argv.index(f); v = c(sys.argv[i+1]); del sys.argv[i:i+2]; return v
    return d

def rollout(env, acts, sp):
    for _ in range(10):
        obs, _ = env.reset(seed=42, options={"eval_setpoint": sp})
        st = env.main_node.get_sim_time()
        if st is not None and st < 0.4:
            break
        print(f"[retry] discarding rollout, simtime={st}", flush=True)
    log = [obs.copy()]
    for a in acts:
        obs, _, term, trunc, _ = env.step(a)
        log.append(obs.copy())
        if term or trunc: break
    return np.array(log, dtype=np.float64)

def main():
    steps = flag("--steps", 400, int)
    out   = flag("--out", "./openloop_noise.npz", str)
    rclpy.init()
    env = gym.make("Autopilot-RL-v0").unwrapped
    acts = np.zeros((steps, 3), dtype=np.float32)
    sp = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    try:
        for _ in range(5):                      # settle the sim clock
            rollout(env, acts[:steps], sp)
        A = rollout(env, acts, sp)
        B = rollout(env, acts, sp)
        n = min(len(A), len(B)); A, B = A[:n], B[:n]
        d_acc = A[:, 6:9] - B[:, 6:9]
        d_ang = A[:, 3:6] - B[:, 3:6]
        print(f"\nOPEN LOOP  n={n} states")
        print(f"  accel  : std={d_acc.std():.6e}   per-axis mean={np.array2string(d_acc.mean(0), precision=6)}")
        print(f"  angvel : std={d_ang.std():.6e}")
        print(f"  max|d| over accel = {np.abs(d_acc).max():.6e}")
        np.savez(out, d_acc=d_acc, d_ang=d_ang, obs_a=A, obs_b=B)
        print(f"  saved {out}")
    finally:
        if hasattr(env, "main_node"): env.main_node.destroy_node()
        env.close(); rclpy.shutdown()

main()
