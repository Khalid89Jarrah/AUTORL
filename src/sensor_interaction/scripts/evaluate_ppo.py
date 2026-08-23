#!/usr/bin/env python3

import os
import gymnasium as gym
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import rclpy
from dataclasses import dataclass
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from sensor_interaction.autorl_world import Auto_RL


def frd_to_flu(v):
    v = np.asarray(v, dtype=np.float32)
    return np.array([v[0], -v[1], -v[2]], dtype=np.float32)


def flu_to_frd(v):
    v = np.asarray(v, dtype=np.float32)
    return np.array([v[0], -v[1], -v[2]], dtype=np.float32)


def plot_tracking(ax, time, actual_frd, desired_frd, eval_setpoint_frd):
    labels = ["Roll (X)", "Pitch (Y)", "Yaw (Z)"]
    for i in range(3):
        ax.plot(time, actual_frd[:, i], label=f"Actual {labels[i]}", linestyle="-")
        ax.plot(time, desired_frd[:, i], label=f"Desired {labels[i]}", linestyle="--")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angular Velocity (rad/s)")
    ax.set_title(f"Tracking FRD Setpoint {eval_setpoint_frd}")
    ax.grid(True)
    ax.legend()


def plot_oscillations(ax, time, actual_frd, eval_setpoint_frd):
    labels = ["Roll (X)", "Pitch (Y)", "Yaw (Z)"]
    for i in range(3):
        ax[i].plot(time, actual_frd[:, i], label=f"Actual {labels[i]}")
        ax[i].axhline(
            y=eval_setpoint_frd[i], linestyle="--", label=f"Setpoint {labels[i]}"
        )
        ax[i].set_xlabel("Time (s)")
        ax[i].set_ylabel("Angular Velocity (rad/s)")
        ax[i].grid(True)
        ax[i].legend()


@dataclass
class AxisMetrics:
    rise_time: float
    overshoot_pct: float
    settling_time: float
    ss_error_abs: float
    IAE: float
    ISE: float
    ITAE: float
    effort_L1: float
    effort_L2: float
    sat_frac: float
    time_in_thresh_frac: float
    # --- Campaign B: added for the reward-ablation study ---
    # Judges the `oscillation` reward term (Eq. 9) on its own objective
    # (shakiness) instead of on tracking error, which cannot see it.
    osc_energy: float


def _rise_time(t, y, y_ss, frac_low=0.1, frac_high=0.9):
    try:
        y0 = y[0]
        lo = y0 + (y_ss - y0) * frac_low
        hi = y0 + (y_ss - y0) * frac_high
        i10 = np.where((y[:-1] - lo) * (y[1:] - lo) <= 0)[0][0] + 1
        i90 = np.where((y[:-1] - hi) * (y[1:] - hi) <= 0)[0][0] + 1
        return float(t[i90] - t[i10])
    except Exception:
        return float("nan")


def _overshoot_pct(y, y_ss):
    if np.isclose(y_ss, 0.0):
        return 0.0
    peak = np.max(y) if y_ss >= 0 else np.min(y)
    return 100.0 * abs(peak - y_ss) / (abs(y_ss) + 1e-9)


def _settling_time(t, y, y_ss, eps=0.05, hold=0.5):
    if len(t) < 2:
        return float("nan")
    dt = float(np.mean(np.diff(t)))
    need = max(1, int(round(hold / dt)))
    within = np.abs(y - y_ss) <= eps
    for k in range(len(within) - need):
        if within[k] and np.all(within[k : k + need]):
            return float(t[k])
    return float("nan")


def _osc_energy(y):
    """Campaign B: mean squared discrete second difference of the tracked
    signal, i.e. the exact same expression as the reward's oscillation
    penalty (Eq. 9: ||omega_t - 2*omega_{t-1} + omega_{t-2}||^2), but
    computed here as a held-out MEASUREMENT rather than a training penalty.

    This is the metric that judges whether removing the `oscillation`
    reward term (ablate="oscillation") actually makes the drone shakier,
    independent of what happens to tracking error. Tracking error cannot
    see this: a policy can chase the setpoint harder and record a LOWER
    tracking error while oscillating MORE.
    """
    if len(y) < 3:
        return float("nan")
    second_diff = y[2:] - 2.0 * y[1:-1] + y[:-2]
    return float(np.mean(second_diff**2))


def compute_axis_metrics(t, y, sp, u, eps=0.05, hold=0.5):
    dt = float(np.mean(np.diff(t))) if len(t) > 1 else 0.01
    e = sp - y
    ss_start = max(0, int(0.9 * len(y)))
    ss_error_abs = float(np.mean(np.abs(e[ss_start:]))) if len(y) else float("nan")
    return AxisMetrics(
        _rise_time(t, y, sp),
        _overshoot_pct(y, sp),
        _settling_time(t, y, sp, eps, hold),
        ss_error_abs,
        float(np.sum(np.abs(e)) * dt),
        float(np.sum(e**2) * dt),
        float(np.sum(t[: len(e)] * np.abs(e)) * dt),
        float(np.sum(np.abs(u)) * dt),
        float(np.sum(u**2) * dt),
        float(np.mean(np.isclose(np.abs(u), 1.0, atol=1e-6))) if len(u) else 0.0,
        float(np.mean(np.abs(e) <= eps)) if len(e) else 0.0,
        _osc_energy(y),
    )


def evaluate_ppo(
    env,
    model,
    eval_setpoints_frd,
    evaluation_time=30.0,
    simulation_step_time=0.004,
    settle_steps=3,
    csv_path="metrics_ppo.csv",
):
    time_steps = int(evaluation_time / simulation_step_time)
    base_time = np.linspace(0, evaluation_time, time_steps, dtype=np.float32)
    plot_payloads = []
    all_rows = []

    for i, sp_frd in enumerate(eval_setpoints_frd):
        sp_flu = frd_to_flu(sp_frd)
        obs, _ = env.reset(options={"eval_setpoint": sp_flu})

        # Warm-up
        for _ in range(max(0, settle_steps)):
            obs, _, _, _, _ = env.step(np.zeros(3, dtype=np.float32))

        t_hist, desired_hist, actual_hist, action_hist = [], [], [], []
        print(
            f"\n=== PPO Setpoint {i+1}: FRD={sp_frd.tolist()} | FLU={sp_flu.tolist()} ==="
        )

        step = 0
        while step < time_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs_next, _, terminated, truncated, _ = env.step(action)
            meas_frd = flu_to_frd(obs_next[3:6])

            t_hist.append(step * simulation_step_time)
            desired_hist.append(sp_frd.copy())
            actual_hist.append(meas_frd.copy())
            action_hist.append(np.asarray(action, dtype=np.float32).copy())

            obs = obs_next
            step += 1

            if terminated or truncated:
                obs, _ = env.reset(options={"eval_setpoint": sp_flu})
                continue

        if len(actual_hist) == 0:
            print(f"Skipping setpoint {sp_frd}: no samples collected.")
            continue

        t_hist = np.array(t_hist)
        desired_hist = np.array(desired_hist)
        actual_hist = np.array(actual_hist)
        action_hist = np.array(action_hist)

        rows = []
        for ax in range(3):
            m = compute_axis_metrics(
                t_hist, actual_hist[:, ax], sp_frd[ax], action_hist[:, ax]
            )
            rows.append(
                {
                    "setpoint_index": i + 1,
                    "axis": ["Roll_X", "Pitch_Y", "Yaw_Z"][ax],
                    "sp_roll": float(sp_frd[0]),
                    "sp_pitch": float(sp_frd[1]),
                    "sp_yaw": float(sp_frd[2]),
                    "rise_time_s": m.rise_time,
                    "overshoot_pct": m.overshoot_pct,
                    "settling_time_s": m.settling_time,
                    "ss_error_rad_s": m.ss_error_abs,
                    "IAE": m.IAE,
                    "ISE": m.ISE,
                    "ITAE": m.ITAE,
                    "effort_L1": m.effort_L1,
                    "effort_L2": m.effort_L2,
                    "sat_frac": m.sat_frac,
                    "time_in_thresh_frac": m.time_in_thresh_frac,
                    # Campaign B: judges the `oscillation` reward term (Eq. 9)
                    "osc_energy": m.osc_energy,
                }
            )

        df = pd.DataFrame(rows)
        all_rows.append(df)
        print(df.to_string(index=False, float_format=lambda x: f"{x:,.6g}"))

        plot_payloads.append(
            (base_time.copy(), actual_hist.copy(), desired_hist.copy(), sp_frd.copy())
        )

    if all_rows:
        out = pd.concat(all_rows, ignore_index=True)

        # Compute averages
        numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
        for col in ["sp_roll", "sp_pitch", "sp_yaw"]:
            if col in numeric_cols:
                numeric_cols.remove(col)

        averages = out[numeric_cols].mean()

        blank_row = {col: "" for col in out.columns}

        avg_row = {col: "" for col in out.columns}
        for col in numeric_cols:
            avg_row[col] = averages[col]

        avg_row["setpoint_index"] = "AVERAGE"
        avg_row["axis"] = ""
        avg_row["sp_roll"] = ""
        avg_row["sp_pitch"] = ""
        avg_row["sp_yaw"] = ""

        out = pd.concat([out, pd.DataFrame([blank_row, avg_row])], ignore_index=True)

        if "highlight" in out.columns:
            out = out.drop(columns=["highlight"])

        out.to_csv(csv_path, index=False)

        print("\n=== PPO Controller Average Performance Metrics ===")
        for col in numeric_cols:
            try:
                # osc_energy is typically ~1e-5 to 1e-6: use scientific
                # notation so it doesn't print as 0.000000 (handoff §7.3).
                if col == "osc_energy":
                    print(f"{col:25s}: {averages[col]:.6e}")
                else:
                    print(f"{col:25s}: {averages[col]:.6f}")
            except Exception:
                pass
        print("=================================================\n")

    if plot_payloads:
        all_vals = np.vstack([np.vstack((a, d)) for (_, a, d, _) in plot_payloads])
        y_min, y_max = np.min(all_vals), np.max(all_vals)
        margin = 0.05 * (y_max - y_min)
        y_min -= margin
        y_max += margin

        # Tracking plots
        fig_tr, ax_tr = plt.subplots(2, 2, figsize=(14, 10))
        ax_tr = ax_tr.flatten()

        for idx, (t_base, actual_hist, desired_hist, sp_frd) in enumerate(
            plot_payloads
        ):
            ax = ax_tr[idx]
            t = t_base[: len(actual_hist)]
            plot_tracking(ax, t, actual_hist, desired_hist[: len(actual_hist)], sp_frd)
            ax.set_ylim(y_min, y_max)

        plt.tight_layout()
        tracking_path = os.path.abspath("ppo_tracking.png")
        fig_tr.savefig(tracking_path, dpi=150)

        # Oscillations plots
        n = len(plot_payloads)
        fig_osc, ax_osc = plt.subplots(n, 3, figsize=(15, 3 * n))
        if n == 1:
            ax_osc = [ax_osc]

        for idx, (t_base, actual_hist, _, sp_frd) in enumerate(plot_payloads):
            plot_oscillations(
                ax_osc[idx], t_base[: len(actual_hist)], actual_hist, sp_frd
            )

        plt.tight_layout()
        osc_path = os.path.abspath("ppo_oscillations.png")
        fig_osc.savefig(osc_path, dpi=150)
        print(f"\nSaved plots:\n  {tracking_path}\n  {osc_path}")


def main(args=None):
    rclpy.init(args=args)
    env = gym.make("Autopilot-RL-v0")
    check_env(env)
    from rcl_interfaces.msg import ParameterDescriptor

    tmp_node = rclpy.create_node("ppo_eval_param_loader")
    desc = ParameterDescriptor(dynamic_typing=True)
    model_path = tmp_node.declare_parameter("model_path", None, desc).value
    if not model_path:
        raise RuntimeError("ERROR: You must pass model_path:=/path/to/model.zip")
    tmp_node.destroy_node()

    model = PPO.load(model_path, device="cpu")

    setpoints = [
        np.array([-0.5, 0.3, -0.4], dtype=np.float32),
        np.array([0.1, -0.2, 0.6], dtype=np.float32),
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        np.array([-0.3, -0.1, 0.2], dtype=np.float32),
    ]

    try:
        evaluate_ppo(env, model, setpoints)
    finally:
        if hasattr(env.unwrapped, "main_node"):
            env.unwrapped.main_node.destroy_node()
        env.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
