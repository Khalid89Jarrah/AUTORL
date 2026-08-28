#!/usr/bin/env python3
"""
Unified evaluator for AutoRL / Campaign C.

Rewritten from the original evaluate_ppo.py (4 hardcoded setpoints,
PPO.load() hardcoded, no CLI flags) to fix two problems discovered directly:

1. NO CONTROLLER DISPATCH. The original script always called PPO.load(),
   regardless of what model was actually being evaluated. Pointing it at a
   TD3 checkpoint crashed: SB3 built a TD3Policy under PPO's _setup_model(),
   which passes PPO-only kwargs (use_sde) that TD3Policy.__init__() does not
   accept -> TypeError, process death, no CSV/NPZ ever written.
   (Traceback observed: evaluate_ppo.py:298, TD3Policy.__init__() got an
   unexpected keyword argument 'use_sde'.)

2. evalcampainc.sbatch sets AUTORL_CONTROLLER / AUTORL_TAG / AUTORL_EVAL_OUT
   env vars and expects this script to read them. The original script never
   read any of the three - they were silently ignored.

Also corrects, since they were already known issues on this branch:
- dt was hardcoded to 0.01s / 0.004s in different places; here it is read
  from the world SDF's <max_step_size> so it can never disagree with the
  simulator again.
- Only 4 fixed setpoints were evaluated per run, not comparable across
  seeds/algorithms in a statistically meaningful way. Replaced with a
  shared, fixed-seed 31-setpoint bank (30 from the training distribution
  N(0, 0.3) plus zero) so every controller/seed/checkpoint is scored on
  the exact same setpoints.

Usage (inside the container):
  ros2 launch sensor_interaction node_launch.py algorithm:=ppo gui:=false \
       mode:=evaluate model_path:=/path/model.zip

Controller/output flags, read from --flag or environment variable:
  --controller ppo|td3|sac|pid     (env: AUTORL_CONTROLLER, default ppo)
  --tag        <string>            (env: AUTORL_TAG, default: model filename)
  --out-dir    <path>              (env: AUTORL_EVAL_OUT, default: cwd)
  --n-setpoints <int>               (env: AUTORL_N_SETPOINTS, default 31)
  --model      <path>              (env: AUTORL_MODEL; model_path:= ROS param
                                     is also accepted and takes priority)
"""
import os
import sys
import re
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import rclpy
import gymnasium as gym

from stable_baselines3 import PPO, TD3, SAC

from sensor_interaction.autorl_world import Auto_RL
from sensor_interaction.MainNode import MainNode

# Fixed RNG seed for the evaluation setpoint bank. Every controller, every
# training seed and every checkpoint sees exactly these setpoints, so
# comparisons across them are paired, not just averaged.
SETPOINT_RNG_SEED = 12345

AXES = ("Roll_X", "Pitch_Y", "Yaw_Z")


# ---------------------------------------------------------------------------
# CLI / environment helpers (stripped from argv before rclpy.init sees them)
# ---------------------------------------------------------------------------
def _take_flag(flag, env_var, default):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            v = sys.argv[i + 1]
            del sys.argv[i : i + 2]
            return v
    return os.environ.get(env_var, default)


# ---------------------------------------------------------------------------
# Simulation step size: read it, never assume it
# ---------------------------------------------------------------------------
def resolve_dt():
    """Read <max_step_size> from the installed world SDF.

    Previously hardcoded (0.01s in one place, 0.004s in another on this
    branch) - both were assumptions. Reading it directly means this value
    can never silently disagree with the simulator again.
    """
    override = os.environ.get("AUTORL_DT")
    if override:
        return float(override)

    patterns = [
        "/opt/autorl_ws/install/sensor_interaction/share/sensor_interaction/"
        "resources/autorl_drone/autorl_drone.sdf",
        "/opt/autorl_ws/src/sensor_interaction/resources/autorl_drone/autorl_drone.sdf",
    ]
    for p in patterns:
        if os.path.exists(p):
            m = re.search(
                r"<max_step_size>\s*([0-9.eE+-]+)\s*</max_step_size>", open(p).read()
            )
            if m:
                dt = float(m.group(1))
                print(f"[eval] dt = {dt} s  (read from {p})")
                return dt
    raise RuntimeError(
        "Could not read <max_step_size> from the world SDF. "
        "Set AUTORL_DT explicitly if the world lives elsewhere."
    )


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------
def flu_to_frd(v):
    v = np.asarray(v, dtype=np.float32)
    return np.array([v[0], -v[1], -v[2]], dtype=np.float32)


def frd_to_flu(v):
    v = np.asarray(v, dtype=np.float32)
    return np.array([v[0], -v[1], -v[2]], dtype=np.float32)


# ---------------------------------------------------------------------------
# PID baseline (gains unchanged from the original evaluate_pid.py main())
# ---------------------------------------------------------------------------
class PIDRateController:
    def __init__(
        self,
        kp=(0.11, 0.14, 0.30),
        ki=(0.18, 0.12, 0.15),
        kd=(0.015, 0.006, 0.00),
        u_min=-1.0,
        u_max=1.0,
        integ_limit=(0.5, 0.4, 0.35),
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


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
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


def _overshoot_pct(y, y_ss, min_sp=0.05):
    """Percentage overshoot relative to the setpoint.

    Returns NaN when |setpoint| < min_sp: dividing by a near-zero setpoint
    makes the percentage meaningless (a tiny transient on a ~0 setpoint can
    read as hundreds of percent). NaN is skipped by pandas .mean(), so only
    axes with a meaningful setpoint contribute. 0.05 rad/s matches the same
    threshold used by time_in_thresh_frac and the environment's own
    stability check.
    """
    if abs(y_ss) < min_sp:
        return float("nan")
    peak = np.max(y) if y_ss >= 0 else np.min(y)
    return 100.0 * abs(peak - y_ss) / (abs(y_ss) + 1e-9)


def _settling_time(t, y, y_ss, eps=0.05, hold=0.5):
    if len(t) < 2:
        return float("nan")
    dt = float(np.mean(np.diff(t)))
    need = max(1, int(round(hold / dt)))
    within = np.abs(y - y_ss) <= eps
    if len(within) <= need:
        return float("nan")
    for k in range(len(within) - need):
        if within[k] and np.all(within[k : k + need]):
            return float(t[k])
    return float("nan")


def _osc_energy(y):
    """Mean squared second difference: the discrete surrogate for angular
    acceleration that the oscillation reward term (paper Eq. 9) penalizes.
    Judging that reward term by tracking error alone says nothing about
    whether it did what it was designed to do."""
    if len(y) < 3:
        return float("nan")
    d2 = y[2:] - 2.0 * y[1:-1] + y[:-2]
    return float(np.mean(d2**2))


def compute_axis_metrics(t, y, sp, u, dt, eps=0.05, hold=0.5):
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


# ---------------------------------------------------------------------------
# Setpoint bank
# ---------------------------------------------------------------------------
def build_setpoints(n):
    """n-1 setpoints drawn from the training distribution N(0, 0.3) plus the
    zero setpoint (hover regulation). Fixed seed => identical for every
    controller, seed, and checkpoint evaluated."""
    rng = np.random.default_rng(SETPOINT_RNG_SEED)
    sps = [np.zeros(3, dtype=np.float32)]
    while len(sps) < n:
        sps.append(rng.normal(0.0, 0.3, size=3).astype(np.float32))
    return sps


# ---------------------------------------------------------------------------
# Rollout: one clean episode per setpoint, no mid-episode resets
# ---------------------------------------------------------------------------
def run_episode(env, sp_frd, dt, controller, model=None, pid=None, max_steps=100000):
    sp_flu = frd_to_flu(sp_frd)
    obs, _ = env.reset(options={"eval_setpoint": sp_flu})
    if controller == "pid":
        pid.reset(measurement=flu_to_frd(obs[3:6]))

    t_hist, y_hist, u_hist = [], [], []
    step = 0
    end_reason = "max_steps"

    while step < max_steps:
        # The action space is already in the PX4 FRD convention; it is fed
        # straight into the FRD control-allocation mixer, unconverted.
        if controller == "pid":
            meas_frd = flu_to_frd(obs[3:6])
            err = sp_frd - meas_frd
            action, _ = pid(err, meas_frd, dt)
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, _, terminated, truncated, _ = env.step(action)

        # The measured angular velocity comes from the IMU in FLU, so it
        # does need converting for reporting. The action does not.
        t_hist.append(step * dt)
        y_hist.append(flu_to_frd(obs[3:6]))
        u_hist.append(np.asarray(action, dtype=np.float32).copy())
        step += 1

        if terminated:
            end_reason = "terminated"
            break
        if truncated:
            end_reason = "truncated"
            break

    return (
        np.asarray(t_hist, dtype=np.float64),
        np.asarray(y_hist, dtype=np.float64),
        np.asarray(u_hist, dtype=np.float64),
        end_reason,
    )


def plot_tracking(ax, t, actual, desired, sp):
    labels = ["Roll (X)", "Pitch (Y)", "Yaw (Z)"]
    for i in range(3):
        ax.plot(t, actual[:, i], label=f"Actual {labels[i]}")
        ax.plot(t, desired[:, i], label=f"Desired {labels[i]}", linestyle="--")
    ax.set_title(f"Tracking FRD Setpoint {np.round(sp, 3)}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angular Velocity (rad/s)")
    ax.grid(True)
    ax.legend(fontsize=7)


# ---------------------------------------------------------------------------
def evaluate(env, controller, model, pid, setpoints, dt, tag, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows, raw, plot_payloads = [], {}, []

    for i, sp in enumerate(setpoints):
        t, y, u, end_reason = run_episode(env, sp, dt, controller, model=model, pid=pid)
        if len(t) < 3:
            print(f"[eval] setpoint {i} produced only {len(t)} samples - skipped")
            continue

        raw[f"sp{i}_t"] = t
        raw[f"sp{i}_y"] = y
        raw[f"sp{i}_u"] = u
        raw[f"sp{i}_sp"] = sp

        for a, name in enumerate(AXES):
            m = compute_axis_metrics(t, y[:, a], float(sp[a]), u[:, a], dt)
            rows.append(
                {
                    "tag": tag,
                    "controller": controller,
                    "setpoint_index": i,
                    "axis": name,
                    "sp_roll": float(sp[0]),
                    "sp_pitch": float(sp[1]),
                    "sp_yaw": float(sp[2]),
                    "sp_value": float(sp[a]),
                    "n_steps": len(t),
                    "duration_s": float(t[-1]),
                    "end_reason": end_reason,
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
                    "osc_energy": m.osc_energy,
                }
            )
        print(
            f"[eval] setpoint {i:2d} {np.round(sp,3)} "
            f"steps={len(t):4d} ({t[-1]:.2f} s) end={end_reason}"
        )
        if i < 4:  # keep the tracking plot readable; only first 4 setpoints
            plot_payloads.append((t.copy(), y.copy(), np.tile(sp, (len(t), 1)), sp.copy()))

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, f"metrics_{tag}.csv")
    npz_path = os.path.join(out_dir, f"raw_{tag}.npz")
    df.to_csv(csv_path, index=False)
    np.savez_compressed(npz_path, dt=dt, **raw)

    num = df.select_dtypes(include=[np.number]).drop(
        columns=["setpoint_index", "sp_value", "sp_roll", "sp_pitch", "sp_yaw"],
        errors="ignore",
    )
    print(f"\n=== {tag} ({controller}) mean over {len(setpoints)} setpoints x 3 axes ===")
    print(num.mean().to_string())
    print(f"\nwrote {csv_path}\nwrote {npz_path}\n")

    if plot_payloads:
        try:
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))
            axs = axs.flatten()
            for idx, (t, y, d, sp) in enumerate(plot_payloads):
                plot_tracking(axs[idx], t, y, d, sp)
            plt.tight_layout()
            png_path = os.path.join(out_dir, f"{tag}_tracking.png")
            fig.savefig(png_path, dpi=150)
            plt.close(fig)
            print(f"wrote {png_path}")
        except Exception as e:
            print(f"[eval] plotting failed (non-fatal): {e}")

    return df


# ---------------------------------------------------------------------------
def main(args=None):
    controller = _take_flag("--controller", "AUTORL_CONTROLLER", "ppo").lower()
    tag = _take_flag("--tag", "AUTORL_TAG", "")
    out_dir = _take_flag("--out-dir", "AUTORL_EVAL_OUT", os.getcwd())
    n_sp = int(_take_flag("--n-setpoints", "AUTORL_N_SETPOINTS", "31"))

    if controller not in ("ppo", "td3", "sac", "pid"):
        print(f"ERROR: unknown controller '{controller}'. Valid: ppo, td3, sac, pid")
        sys.exit(1)

    rclpy.init(args=args)
    dt = resolve_dt()
    env = gym.make("Autopilot-RL-v0")

    model = None
    pid = None
    if controller == "pid":
        pid = PIDRateController()
        if not tag:
            tag = "pid"
    else:
        from rcl_interfaces.msg import ParameterDescriptor

        tmp = rclpy.create_node("eval_param_loader")
        desc = ParameterDescriptor(dynamic_typing=True)
        model_path = tmp.declare_parameter("model_path", None, desc).value
        tmp.destroy_node()
        if not model_path:
            model_path = _take_flag("--model", "AUTORL_MODEL", "")
        if not model_path:
            print("ERROR: pass model_path:=/path/model.zip (or --model)")
            sys.exit(1)

        cls = {"ppo": PPO, "td3": TD3, "sac": SAC}[controller]
        print(f"[eval] loading {controller.upper()} model from {model_path}")
        model = cls.load(model_path, device="cpu")
        if not tag:
            tag = os.path.splitext(os.path.basename(model_path))[0]

    setpoints = build_setpoints(n_sp)
    print(f"[eval] controller={controller} tag={tag} setpoints={len(setpoints)} dt={dt}")

    try:
        evaluate(env, controller, model, pid, setpoints, dt, tag, out_dir)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise
    finally:
        if hasattr(env.unwrapped, "main_node"):
            env.unwrapped.main_node.destroy_node()
        env.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
