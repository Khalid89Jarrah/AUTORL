#!/usr/bin/env python3
import argparse, time
import numpy as np
import torch
from stable_baselines3 import PPO
import tflite_runtime.interpreter as tflite


def sb3_policy_output(policy, obs_np):
    """Return final PPO output (includes tanh)."""
    with torch.no_grad():
        x = torch.from_numpy(obs_np).float()
        x = policy.mlp_extractor.policy_net(x)
        x = policy.action_net(x)
        x = torch.tanh(x)  # Match PPO export with tanh
        return x.numpy()


def percentile(a, p):
    return float(np.percentile(a, p)) if len(a) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="Path to PPO model .zip")
    ap.add_argument("--tflite", required=True, help="Path to .tflite model")
    ap.add_argument("--n", type=int, default=10000, help="Number of random samples")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)

    print("Loading PPO model (CPU, no env needed)…")
    model = PPO.load(args.zip, device="cpu", print_system_info=False)
    policy = model.policy
    obs_space = model.observation_space

    print("Loading TFLite model…")
    interpreter = tflite.Interpreter(model_path=args.tflite)
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]

    diffs = []
    tflite_lat_us = []

    # Sampling
    def sample_uniform():
        if np.all(np.isfinite(obs_space.low)) and np.all(np.isfinite(obs_space.high)):
            return np.random.uniform(obs_space.low, obs_space.high).astype(np.float32)
        return np.random.normal(0, 1, size=obs_space.shape).astype(np.float32)

    def sample_near_hover():
        center = np.zeros_like(obs_space.sample(), dtype=np.float32)
        if np.all(np.isfinite(obs_space.low)) and np.all(np.isfinite(obs_space.high)):
            center = ((obs_space.high + obs_space.low) / 2.0).astype(np.float32)
            scale = 0.05 * (obs_space.high - obs_space.low)
            noise = (
                np.random.normal(0, 1, size=obs_space.shape).astype(np.float32) * scale
            )
            return center + noise
        noise = np.random.normal(0, 0.1, size=obs_space.shape).astype(np.float32)
        return center + noise

    def sample_edges():
        if not (
            np.all(np.isfinite(obs_space.low)) and np.all(np.isfinite(obs_space.high))
        ):
            return []
        lo, hi = obs_space.low.astype(np.float32), obs_space.high.astype(np.float32)
        samples = []
        for _ in range(16):
            mask = np.random.randint(0, 2, size=lo.shape, dtype=bool)
            samples.append(np.where(mask, lo, hi))
        return samples

    # Dataset
    N = args.n
    n_uniform = int(0.8 * N)
    n_hover = int(0.15 * N)
    n_edges = min(int(0.05 * N), 256)

    obs_list = [sample_uniform() for _ in range(n_uniform)]
    obs_list += [sample_near_hover() for _ in range(n_hover)]
    obs_list += sample_edges()[:n_edges]
    while len(obs_list) < N:
        obs_list.append(sample_uniform())

    obs_arr = np.stack(obs_list, axis=0).astype(np.float32)

    # Run
    for i in range(N):
        obs = obs_arr[i : i + 1]
        # SB3 forward (includes tanh)
        sb3_out = sb3_policy_output(policy, obs)

        # TFLite forward
        start = time.perf_counter_ns()
        interpreter.set_tensor(in_det["index"], obs)
        interpreter.invoke()
        tfl_out = interpreter.get_tensor(out_det["index"])
        dt_us = (time.perf_counter_ns() - start) / 1000.0
        tflite_lat_us.append(dt_us)

        diffs.append(np.abs(tfl_out - sb3_out))

    # Metrics
    diffs = np.stack(diffs, axis=0)
    mae = np.mean(diffs, axis=0)
    rmse = np.sqrt(np.mean(diffs**2, axis=0))
    maxabs = np.max(diffs, axis=0)
    p99 = np.percentile(diffs, 99, axis=0)
    tol1 = np.mean(diffs <= 1e-3)
    tol2 = np.mean(diffs <= 1e-2)

    print("\n COMPARISON (PPO vs TFLite) ")
    print(f"MAE per dim     : {mae}")
    print(f"RMSE per dim    : {rmse}")
    print(f"Max |err| per dim: {maxabs}")
    print(f"p99 |err| per dim: {p99}")
    print(f"% within 1e-3   : {100 * tol1:.2f}%")
    print(f"% within 1e-2   : {100 * tol2:.2f}%")

    # Latency
    tlat = np.array(tflite_lat_us)
    print("\n TFLite latency (µs) ")
    print(
        f"mean: {tlat.mean():.2f}, p95: {percentile(tlat,95):.2f}, p99: {percentile(tlat,99):.2f}, min: {tlat.min():.2f}, max: {tlat.max():.2f}"
    )

    # Summary
    print("\n SUMMARY ")
    mean_mae = float(np.mean(mae))
    mean_latency = float(np.mean(tlat))

    if mean_mae < 0.005:
        quality = "EXCELLENT"
    elif mean_mae < 0.015:
        quality = "GOOD"
    elif mean_mae < 0.03:
        quality = "MODERATE"
    else:
        quality = "POOR"

    if mean_latency < 100:
        speed = "FAST"
    elif mean_latency < 1000:
        speed = "OK"
    else:
        speed = "SLOW"

    print(f"Model equivalence quality: {quality} ({mean_mae*100:.2f}% mean diff)")
    print(f"Inference speed: {speed} ({mean_latency:.2f} µs avg)")
    if quality in ("EXCELLENT", "GOOD") and speed in ("FAST", "OK"):
        print("Conversion successful — ready for deployment.")
    else:
        print("Conversion usable but may need better quantization calibration.")


if __name__ == "__main__":
    main()
