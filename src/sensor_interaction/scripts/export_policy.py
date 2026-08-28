#!/usr/bin/env python3
"""
Policy export and inference benchmark.

Answers R2-5: the paper claims embedded compatibility on the strength of a
parameter count alone. This exports the trained actor and measures what can
actually be measured on this machine -- inference latency, exported file size,
parameter count, multiply-accumulate count and fp32 weight footprint.

It does NOT claim anything about a flight controller board. Running on real
hardware stays declared future work; this script produces the numbers that
replace the bare parameter count, and nothing more.

No ROS, no Gazebo.

    python3 export_policy.py --model /opt/autorl_ws/models/ppo_model_ref_seed0.zip \
                             --out ./embedded_report

ON THE PARAMETER COUNT
----------------------
The paper reports 10,628. Stable-Baselines3's MlpPolicy for a continuous action
space also carries a `log_std` parameter, one entry per action dimension (3 for
this task). This script prints both totals so the paper can state which one it
means:  actor+critic weights, and actor+critic weights + log_std.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from stable_baselines3 import PPO


class ActorWrapper(torch.nn.Module):
    """Deterministic actor: observation -> mean action.

    This is the graph an embedded target would run. The critic and the
    stochastic sampling head are training-time only.
    """

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs):
        features = self.policy.extract_features(obs)
        if isinstance(features, tuple):
            latent_pi = self.policy.mlp_extractor.forward_actor(features[0])
        else:
            latent_pi = self.policy.mlp_extractor.forward_actor(features)
        return self.policy.action_net(latent_pi)


def linear_macs(module):
    """Multiply-accumulate count for every Linear layer in a module."""
    total = 0
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            total += m.in_features * m.out_features
    return total


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def benchmark(fn, obs, warmup, iters):
    for _ in range(warmup):
        fn(obs)
    samples = np.empty(iters)
    for i in range(iters):
        t0 = time.perf_counter()
        fn(obs)
        samples[i] = (time.perf_counter() - t0) * 1e3  # ms
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="./embedded_report")
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--no-onnx", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(1)  # single-thread: the embedded-relevant figure

    model = PPO.load(args.model, device="cpu")
    policy = model.policy.eval()

    obs_dim = int(np.prod(model.observation_space.shape))
    act_dim = int(np.prod(model.action_space.shape))
    obs = torch.zeros(1, obs_dim, dtype=torch.float32)

    actor = ActorWrapper(policy).eval()
    with torch.no_grad():
        out = actor(obs)
    assert out.shape[-1] == act_dim, f"actor output {out.shape} != {act_dim}"

    # --- parameter accounting -------------------------------------------------
    named = dict(policy.named_parameters())
    log_std_n = named["log_std"].numel() if "log_std" in named else 0
    total_all = count_params(policy)
    total_wo_logstd = total_all - log_std_n

    actor_params = (
        count_params(policy.mlp_extractor.policy_net)
        + count_params(policy.action_net)
        + count_params(policy.features_extractor)
    )
    critic_params = count_params(policy.mlp_extractor.value_net) + count_params(
        policy.value_net
    )

    actor_macs = linear_macs(policy.mlp_extractor.policy_net) + linear_macs(
        policy.action_net
    )

    # --- export ---------------------------------------------------------------
    ts_path = os.path.join(args.out, "actor_traced.pt")
    with torch.no_grad():
        traced = torch.jit.trace(actor, obs)
    traced.save(ts_path)

    onnx_path = os.path.join(args.out, "actor.onnx")
    onnx_ok, onnx_err = False, None
    if not args.no_onnx:
        try:
            torch.onnx.export(
                actor,
                obs,
                onnx_path,
                input_names=["observation"],
                output_names=["action"],
                opset_version=17,
                dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
            )
            onnx_ok = True
        except Exception as e:  # export is optional; report, do not crash
            onnx_err = str(e)

    # --- latency --------------------------------------------------------------
    with torch.no_grad():
        eager = benchmark(lambda x: actor(x), obs, args.warmup, args.iters)
        jitted = benchmark(lambda x: traced(x), obs, args.warmup, args.iters)

    def stats(s):
        return {
            "mean_ms": float(s.mean()),
            "std_ms": float(s.std(ddof=1)),
            "p50_ms": float(np.percentile(s, 50)),
            "p99_ms": float(np.percentile(s, 99)),
            "max_ms": float(s.max()),
        }

    report = {
        "model": os.path.abspath(args.model),
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "params": {
            "actor": actor_params,
            "critic": critic_params,
            "log_std": log_std_n,
            "total_excluding_log_std": total_wo_logstd,
            "total_including_log_std": total_all,
        },
        "actor_macs_per_inference": actor_macs,
        "actor_fp32_weight_bytes": actor_params * 4,
        "exported_file_bytes": {
            "torchscript": os.path.getsize(ts_path),
            "onnx": os.path.getsize(onnx_path) if onnx_ok else None,
        },
        "onnx_export_ok": onnx_ok,
        "onnx_error": onnx_err,
        "latency_single_thread": {
            "eager": stats(eager),
            "torchscript": stats(jitted),
        },
        "host": os.uname()[1],
        "torch_version": torch.__version__,
        "note": (
            "Measured on the evaluation host with torch threads pinned to 1. "
            "These are not flight-controller numbers."
        ),
    }

    with open(os.path.join(args.out, "embedded_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"actor params            : {actor_params}")
    print(f"critic params           : {critic_params}")
    print(f"log_std params          : {log_std_n}")
    print(f"total (excl. log_std)   : {total_wo_logstd}")
    print(f"total (incl. log_std)   : {total_all}")
    print(f"actor MACs / inference  : {actor_macs}")
    print(f"actor fp32 weights      : {actor_params * 4} bytes")
    print(f"torchscript file        : {os.path.getsize(ts_path)} bytes")
    if onnx_ok:
        print(f"onnx file               : {os.path.getsize(onnx_path)} bytes")
    else:
        print(f"onnx export             : FAILED ({onnx_err})")
    print(
        "latency (torchscript)   : "
        f"{jitted.mean():.4f} ms mean, {np.percentile(jitted,99):.4f} ms p99"
    )
    print(f"\nWritten to {args.out}/embedded_report.json")


if __name__ == "__main__":
    main()
