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
from sensor_interaction.autorl_world import Auto_RL
from stable_baselines3.common.env_checker import check_env


def frd_to_flu(v):
    return np.array([v[0], -v[1], -v[2]], dtype=np.float32)


def flu_to_frd(v):
    return np.array([v[0], -v[1], -v[2]], dtype=np.float32)


class PIDRateController:
    def __init__(
        self,
        kp=(0.8, 0.8, 0.5),
        ki=(0.1, 0.1, 0.05),
        kd=(0.02, 0.02, 0.01),
        u_min=-1.0,
        u_max=1.0,
        integ_limit=(0.5, 0.5, 0.3),
    ):
        self.kp = np.array(kp)
        self.ki = np.array(ki)
        self.kd = np.array(kd)
        self.u_min = u_min
        self.u_max = u_max
        self.integ_limit = np.array(integ_limit)
        self._integral = np.zeros(3)
        self._prev_meas = np.zeros(3)
        self._first = True

    def reset(self, measurement=None):
        self._integral.fill(0.0)
        if measurement is not None:
            self._prev_meas = measurement
            self._first = False
        else:
            self._first = True
            self._prev_meas.fill(0.0)

    def __call__(self, error_frd, meas_frd, dt):
        e = np.array(error_frd)
        y = np.array(meas_frd)
        self._integral = np.clip(
            self._integral + e * dt, -self.integ_limit, self.integ_limit
        )
        dy = np.zeros_like(y) if self._first or dt <= 0 else (y - self._prev_meas) / dt
        self._prev_meas = y
        self._first = False
        u_raw = self.kp * e + self.ki * self._integral - self.kd * dy
        return np.clip(u_raw, self.u_min, self.u_max), u_raw


def plot_tracking(ax, t, actual, desired, sp):
    labels = ["Roll (X)", "Pitch (Y)", "Yaw (Z)"]
    for i in range(3):
        ax.plot(t, actual[:, i], label=f"Actual {labels[i]}")
        ax.plot(t, desired[:, i], label=f"Desired {labels[i]}", linestyle="--")
    ax.set_title(f"Tracking FRD Setpoint {sp}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angular Velocity (rad/s)")
    ax.grid()
    ax.legend()


def plot_oscillations(ax, t, actual, sp, ylim=None):
    labels = ["Roll (X)", "Pitch (Y)", "Yaw (Z)"]
    for i in range(3):
        ax[i].plot(t, actual[:, i], label=f"Actual {labels[i]}")
        ax[i].axhline(y=sp[i], linestyle="--", label=f"Setpoint {labels[i]}")
        ax[i].set_xlabel("Time (s)")
        ax[i].set_ylabel("Angular Velocity (rad/s)")
        ax[i].grid()
        ax[i].legend()
        if ylim is not None:
            ax[i].set_ylim(*ylim)


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


def _rise_time(t, y, y_ss, frac_low=0.1, frac_high=0.9):
    try:
        y0 = y[0]
        lo = y0 + (y_ss - y0) * frac_low
        hi = y0 + (y_ss - y0) * frac_high
        i10 = np.where((y[:-1] - lo) * (y[1:] - lo) <= 0)[0][0] + 1
        i90 = np.where((y[:-1] - hi) * (y[1:] - hi) <= 0)[0][0] + 1
        return float(t[i90] - t[i10])
    except:
        return float("nan")


def _overshoot_pct(y, y_ss):
    return (
        0.0
        if np.isclose(y_ss, 0)
        else 100
        * abs((np.max(y) if y_ss >= 0 else np.min(y)) - y_ss)
        / (abs(y_ss) + 1e-9)
    )


def _settling_time(t, y, y_ss, eps=0.05, hold=0.5):
    if len(t) < 2:
        return float("nan")
    dt = float(np.mean(np.diff(t)))
    need = max(1, int(round(hold / dt)))
    within = np.abs(y - y_ss) <= eps
    for k in range(0, len(within) - need):
        if within[k] and np.all(within[k : k + need]):
            return float(t[k])
    return float("nan")


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
    )


def evaluate_pid(
    env,
    pid,
    setpoints,
    evaluation_time=30.0,
    simulation_step_time=0.004,
    csv_path="metrics_pid.csv",
):
    time_steps = int(evaluation_time / simulation_step_time)
    plot_payloads, all_rows = [], []
    min_points = 5

    for i, sp_frd in enumerate(setpoints):
        sp_flu = frd_to_flu(sp_frd)
        obs, _ = env.reset(options={"eval_setpoint": sp_flu})
        pid.reset(measurement=flu_to_frd(obs[3:6]))

        t_hist, desired_hist, actual_hist, action_hist = [], [], [], []
        print(
            f"\n=== PID Setpoint {i+1}: FRD={sp_frd.tolist()} | FLU={sp_flu.tolist()} ==="
        )

        step = 0
        while step < time_steps:
            meas_frd = flu_to_frd(obs[3:6])
            err = sp_frd - meas_frd
            u_sat, _ = pid(err, meas_frd, simulation_step_time)

            obs_next, _, terminated, truncated, _ = env.step(u_sat)
            meas_next_frd = flu_to_frd(obs_next[3:6])

            t_hist.append(step * simulation_step_time)
            desired_hist.append(sp_frd.copy())
            actual_hist.append(meas_next_frd.copy())
            action_hist.append(u_sat.copy())

            step += 1
            obs = obs_next

            if terminated or truncated:
                obs, _ = env.reset(options={"eval_setpoint": sp_flu})
                pid.reset(measurement=flu_to_frd(obs[3:6]))
                continue

        if len(actual_hist) < min_points:
            print(f"Skipping setpoint {sp_frd}: only {len(actual_hist)} samples.")
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
                }
            )

        df = pd.DataFrame(rows)
        all_rows.append(df)
        print(df.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

        plot_payloads.append(
            (t_hist.copy(), actual_hist.copy(), desired_hist.copy(), sp_frd.copy())
        )

    # -----------------------
    # CSV
    # -----------------------
    if all_rows:
        out = pd.concat(all_rows, ignore_index=True)

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

        out = pd.concat([out, pd.DataFrame([blank_row, avg_row])], ignore_index=True)

        out.to_csv(csv_path, index=False)

        print("\n=== PID Controller Average Performance Metrics ===")
        for col in numeric_cols:
            try:
                print(f"{col:25s}: {averages[col]:.6f}")
            except:
                pass
        print("=================================================\n")

    # Plotting
    if plot_payloads:
        all_vals = np.vstack([np.vstack((a, d)) for (_, a, d, _) in plot_payloads])
        y_min, y_max = np.min(all_vals), np.max(all_vals)
        margin = 0.05 * (y_max - y_min)
        y_min -= margin
        y_max += margin

        n = len(plot_payloads)
        fig_tr, ax_tr = plt.subplots(2, 2, figsize=(14, 10))
        ax_tr = ax_tr.flatten()
        fig_osc, ax_osc = plt.subplots(n, 3, figsize=(15, 3 * n))
        if n == 1:
            ax_osc = [ax_osc]

        for idx, (t_hist, actual_hist, desired_hist, sp_frd) in enumerate(
            plot_payloads
        ):
            ax = ax_tr[idx]
            n_points = min(len(t_hist), len(actual_hist), len(desired_hist))
            plot_tracking(
                ax,
                t_hist[:n_points],
                actual_hist[:n_points],
                desired_hist[:n_points],
                sp_frd,
            )
            ax.set_ylim(y_min, y_max)
            plot_oscillations(
                ax_osc[idx], t_hist[:n_points], actual_hist[:n_points], sp_frd, ylim=(y_min, y_max)
            )

        plt.tight_layout()
        track_path = os.path.abspath("pid_tracking.png")
        osc_path = os.path.abspath("pid_oscillations.png")
        fig_tr.savefig(track_path, dpi=150)
        fig_osc.savefig(osc_path, dpi=150)
        print(f"\nSaved plots:\n  {track_path}\n  {osc_path}")


def main(args=None):
    rclpy.init(args=args)
    env = gym.make("Autopilot-RL-v0")
    check_env(env)
    pid = PIDRateController(
        kp=(0.11, 0.14, 0.30),
        ki=(0.18, 0.12, 0.15),
        kd=(0.015, 0.006, 0.00),
        u_min=-1.0,
        u_max=1.0,
        integ_limit=(0.5, 0.4, 0.35),
    )
    setpoints = [
        np.array([-0.5, 0.3, -0.4], dtype=np.float32),
        np.array([0.1, -0.2, 0.6], dtype=np.float32),
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        np.array([-0.3, -0.1, 0.2], dtype=np.float32),
    ]
    print("Starting PID evaluation (headless)…")
    evaluate_pid(env, pid, setpoints)
    env.close()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
