#!/usr/bin/env python3
"""
Divergence distribution analysis (noise-on regime).

Answers R2-14. Section 4.4 of the paper compares ONE measured maximum against
ONE predicted maximum. This script builds an empirical distribution of the
per-pair divergence over all rollout pairs produced by determinism_suite.py and
compares it against a Monte-Carlo null model drawn from the SDF sensor
parameters.

No ROS, no Gazebo. Run it on the .npz files after the fact.

    python3 divergence_stats.py --in ./determinism_out --out ./divergence_report

TWO CORRECTIONS TO THE PAPER'S DERIVATION
-----------------------------------------
1. Only 3 of the 15 observation elements carry sigma_a. The observation is
   [e_w(3), w(3), a(3), u_prev(3), w_hist(3)]: the acceleration block is 3
   elements; the rate-derived blocks carry sigma_w, which is an order of
   magnitude smaller; u_prev is a policy output and carries no sensor noise at
   all. The paper's N = 85 x 15 = 1275 therefore over-counts the elements that
   the sigma_a scaling applies to.

2. The elements are not independent draws. They come from a closed loop in
   which divergence accumulates across steps. The closed-form extreme-value
   estimate assumes independence and cannot be relied on. This script therefore
   reports the Monte-Carlo null as a REFERENCE BAND, and the empirical
   distribution as the actual evidence.

NOTE ON THE sqrt(2) TERM
------------------------
The paper writes E[max|d|] ~= sigma_a * sqrt(4 ln N). Since
sqrt(4 ln N) = sqrt(2) * sqrt(2 ln N), the sqrt(2) from
d ~ N(0, sigma*sqrt(2)) is already folded into that expression; it was not
dropped. Numerically: 1.86e-3 * sqrt(4 ln 1275) = 9.95e-3, which is the value
printed in the paper. The derivation is self-consistent on this point. The
problems with it are (1) and (2) above, not a missing factor.
"""

import argparse
import glob
import json
import os

import numpy as np

# IMU noise parameters as declared in the Gazebo SDF model (paper Section 4.4).
SIGMA_W = 1.87e-4  # rad/s per axis, angular velocity white noise
SIGMA_A = 1.86e-3  # m/s^2 per axis, linear acceleration white noise
SIGMA_BIAS_W = 3.88e-5  # rad/s, dynamic bias
SIGMA_BIAS_A = 6.0e-3  # m/s^2, dynamic bias

# Which sigma applies to which slice of the 15-element observation vector.
# obs = [e_w(0:3), w(3:6), a(6:9), u_prev(9:12), w_hist(12:15)]
OBS_SIGMA = np.concatenate(
    [
        np.full(3, SIGMA_W),  # angular velocity error
        np.full(3, SIGMA_W),  # angular velocity
        np.full(3, SIGMA_A),  # linear acceleration
        np.full(3, 0.0),  # previous control effort: no sensor noise
        np.full(3, SIGMA_W),  # angular velocity history
    ]
)


def monte_carlo_null(n_steps, n_draws, rng):
    """Distribution of max|delta| under white sensor noise ONLY.

    Two independent rollouts differ by the difference of two noise draws, so
    delta ~ N(0, sigma*sqrt(2)) elementwise. Bias drift is added as a single
    per-rollout constant offset, which is how a slowly varying bias appears over
    a short rollout.
    """
    maxima = np.empty(n_draws)
    scale = OBS_SIGMA * np.sqrt(2.0)
    bias_scale = np.concatenate(
        [
            np.full(3, SIGMA_BIAS_W),
            np.full(3, SIGMA_BIAS_W),
            np.full(3, SIGMA_BIAS_A),
            np.full(3, 0.0),
            np.full(3, SIGMA_BIAS_W),
        ]
    ) * np.sqrt(2.0)

    for i in range(n_draws):
        white = rng.normal(0.0, 1.0, size=(n_steps, 15)) * scale
        bias = rng.normal(0.0, 1.0, size=(1, 15)) * bias_scale
        maxima[i] = np.abs(white + bias).max()
    return maxima


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", default="./determinism_out")
    ap.add_argument("--out", dest="out_dir", default="./divergence_report")
    ap.add_argument("--draws", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.in_dir, "*.npz")))
    if not files:
        raise SystemExit(f"No .npz files found in {args.in_dir}")

    rows = []
    for path in files:
        d = np.load(path, allow_pickle=True)
        if "obs_a" not in d:
            continue
        a, b = d["obs_a"], d["obs_b"]
        n = min(len(a), len(b))
        diff = np.abs(a[:n] - b[:n])
        rows.append(
            {
                "file": os.path.basename(path),
                "machine": str(d["machine"]) if "machine" in d else "unknown",
                "noise_tag": str(d["noise_tag"]) if "noise_tag" in d else "unknown",
                "cpu_load": int(d["cpu_load"]) if "cpu_load" in d else 0,
                "steps": int(n),
                "mean_abs": float(diff.mean()),
                "max_abs": float(diff.max()),
                "max_abs_accel_block": float(diff[:, 6:9].max()),
                "max_abs_rate_blocks": float(
                    np.max([diff[:, 0:6].max(), diff[:, 12:15].max()])
                ),
            }
        )

    if not rows:
        raise SystemExit("No usable rollout pairs found.")

    maxima = np.array([r["max_abs"] for r in rows])
    steps_typ = int(np.median([r["steps"] for r in rows]))

    rng = np.random.default_rng(args.seed)
    null = monte_carlo_null(steps_typ, args.draws, rng)

    # Fraction of measured maxima that fall inside the null model's 99th pct.
    null_p99 = float(np.percentile(null, 99))
    inside = float(np.mean(maxima <= null_p99))

    report = {
        "n_pairs": len(rows),
        "steps_per_rollout_median": steps_typ,
        "measured_max_abs": {
            "min": float(maxima.min()),
            "p50": float(np.percentile(maxima, 50)),
            "p95": float(np.percentile(maxima, 95)),
            "max": float(maxima.max()),
            "mean": float(maxima.mean()),
            "std": float(maxima.std(ddof=1)) if len(maxima) > 1 else 0.0,
        },
        "null_model_max_abs": {
            "p50": float(np.percentile(null, 50)),
            "p95": float(np.percentile(null, 95)),
            "p99": null_p99,
            "max": float(null.max()),
        },
        "fraction_of_pairs_within_null_p99": inside,
        "sigma_a_elements": 3,
        "observation_elements": 15,
        "note": (
            "Null model is white sensor noise plus a per-rollout bias offset, "
            "drawn from the SDF parameters. It assumes independence across "
            "elements and steps, which the closed loop does not satisfy; treat "
            "it as a reference band, not a bound."
        ),
    }

    with open(os.path.join(args.out_dir, "divergence_report.json"), "w") as f:
        json.dump({"summary": report, "per_pair": rows}, f, indent=2)

    np.savez(
        os.path.join(args.out_dir, "divergence_distributions.npz"),
        measured_maxima=maxima,
        null_maxima=null,
    )

    print(f"pairs analysed                  : {len(rows)}")
    print(f"steps per rollout (median)      : {steps_typ}")
    print(
        "measured max|d|  p50/p95/max   : "
        f"{np.percentile(maxima,50):.3e} / {np.percentile(maxima,95):.3e} / "
        f"{maxima.max():.3e}"
    )
    print(
        "null model max|d| p50/p95/p99   : "
        f"{np.percentile(null,50):.3e} / {np.percentile(null,95):.3e} / "
        f"{null_p99:.3e}"
    )
    print(f"pairs within null p99           : {inside*100:.1f}%")
    print(f"\nWritten to {args.out_dir}/divergence_report.json")


if __name__ == "__main__":
    main()
