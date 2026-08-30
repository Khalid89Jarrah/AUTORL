#!/usr/bin/env python3
"""
Disturbance evaluation harness.

Answers R2-4: the paper claims disturbance robustness (abstract line 8,
conclusion line 748) with no disturbance experiment. This runs the trained PPO
policy and the PID baseline under three disturbance families on the same four
setpoints used in evaluate_ppo.py / evaluate_pid.py.

MECHANISM -- and what it actually is
------------------------------------
autorl_drone.sdf line 20 already loads:
    gz::sim::systems::ApplyLinkWrench  (gz-sim-apply-link-wrench-system)

Steady wind and gusts are applied as an external FORCE on base_link through
that system. This is a force disturbance, not an aerodynamic wind field: the
Gazebo WindEffects system is NOT loaded in the world and <enable_wind> is set
only on the ground_plane link (line 215), not on the vehicle. Describe it in the
paper as an external force disturbance. Do not call it a simulated wind field.

Mass variation cannot be done through a wrench. Use --emit-mass-variant to write
a modified copy of x500_base/model.sdf with mass and inertia scaled, then point
GZ_SIM_RESOURCE_PATH at the directory containing it and relaunch. The unmodified
values in x500_base/model.sdf are mass 2.0 kg, Ixx = Iyy = 0.02166666666666667,
Izz = 0.04000000000000001 (lines 9-16).

TOPIC DISCOVERY
---------------
The wrench topic name differs between Gazebo releases. This script does not
guess it: it lists topics with `gz topic -l`, picks the one matching 'wrench',
and aborts with the full topic list if none is found.

USAGE
    ros2 launch sensor_interaction node_launch.py \
        algorithm:=ppo gui:=false mode:=disturbance_eval \
        model_path:=/opt/autorl_ws/models/ppo_model_ref_seed0.zip

    # options:
    #   --controller ppo|pid
    #   --case none|wind|gust|all
    #   --force 2.0            steady force magnitude, N
    #   --gust-force 8.0       impulse magnitude, N
    #   --gust-every 250       steps between gusts
    #   --gust-steps 25        gust duration, steps
    #   --dt 0.004
    #   --emit-mass-variant 1.2   write an SDF with mass/inertia x1.2 and exit
"""

import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import rclpy
import gymnasium as gym

from stable_baselines3 import PPO
from sensor_interaction.autorl_world import Auto_RL  # noqa: F401  (registers env)

# evaluate_pid.py is installed alongside this file in lib/sensor_interaction,
# which is not necessarily on sys.path when launched via `ros2 launch`.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_pid import PIDRateController, compute_axis_metrics  # noqa: E402


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


def flu_to_frd(v):
    v = np.asarray(v, dtype=np.float32)
    return np.array([v[0], -v[1], -v[2]], dtype=np.float32)


def find_wrench_topic():
    """Locate the ApplyLinkWrench topic. Aborts loudly rather than guessing."""
    try:
        out = subprocess.run(
            ["gz", "topic", "-l"], capture_output=True, text=True, timeout=20
        )
    except FileNotFoundError:
        raise SystemExit(
            "ERROR: `gz` CLI not found. It is needed to publish the wrench "
            "message. Install gz-tools or run this inside the AutoRL container."
        )
    topics = [t.strip() for t in out.stdout.splitlines() if t.strip()]
    candidates = [t for t in topics if "wrench" in t.lower()]
    if not candidates:
        raise SystemExit(
            "ERROR: no topic matching 'wrench' is advertised.\n"
            "ApplyLinkWrench is declared at autorl_drone.sdf line 20, so either "
            "the world did not load it or this Gazebo release names the endpoint "
            "differently.\nTopics currently advertised:\n  "
            + "\n  ".join(topics)
        )
    # Prefer the non-persistent, non-clear endpoint for one-shot application.
    for t in candidates:
        if t.endswith("/persistent"):
            return t
    for t in candidates:
        if not t.endswith("/clear"):
            return t
    return candidates[0]


FORCE_OFFSET = (0.0, 0.0, 0.0)


def apply_force(topic, force_xyz, entity_name="x500", link_name="base_link"):
    """Publish one EntityWrench on the persistent endpoint.

    FORCE_OFFSET is in the link frame relative to the link origin. base_link's
    CoM sits at the origin, so a zero offset gives a pure translational force
    and no moment -- which is why the rate loop cannot see it. A non-zero
    offset produces tau = offset x F about the CoM.
    """
    fx, fy, fz = (float(v) for v in force_xyz)
    ox, oy, oz = (float(v) for v in FORCE_OFFSET)
    msg = (
        f'entity: {{name: "{entity_name}", type: MODEL}}, '
        f"wrench: {{force: {{x: {fx}, y: {fy}, z: {fz}}}, "
        f"torque: {{x: 0, y: 0, z: 0}}, "
        f"force_offset: {{x: {ox}, y: {oy}, z: {oz}}}}}"
    )
    subprocess.run(
        ["gz", "topic", "-t", topic, "-m", "gz.msgs.EntityWrench", "-p", msg],
        capture_output=True,
        text=True,
        timeout=10,
    )


def clear_force(topic):
    """Clear all persistent wrenches."""
    base = topic[: -len("/persistent")] if topic.endswith("/persistent") else topic
    subprocess.run(
        ["gz", "topic", "-t", base + "/clear", "-m", "gz.msgs.Entity",
         "-p", 'name: "x500", type: MODEL'],
        capture_output=True, text=True, timeout=10,
    )


def emit_mass_variant(scale, src_sdf, dst_dir):
    """Write a copy of x500_base with mass and inertia multiplied by `scale`."""
    os.makedirs(dst_dir, exist_ok=True)
    src_dir = os.path.dirname(src_sdf)
    for name in os.listdir(src_dir):
        s = os.path.join(src_dir, name)
        d = os.path.join(dst_dir, name)
        if os.path.isfile(s):
            shutil.copy2(s, d)

    dst_sdf = os.path.join(dst_dir, os.path.basename(src_sdf))
    tree = ET.parse(dst_sdf)
    root = tree.getroot()

    changed = []
    for link in root.iter("link"):
        if link.get("name") != "base_link":
            continue
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass = inertial.find("mass")
        if mass is not None:
            old = float(mass.text)
            mass.text = repr(old * scale)
            changed.append(("mass", old, old * scale))
        inertia = inertial.find("inertia")
        if inertia is not None:
            for key in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz"):
                el = inertia.find(key)
                if el is not None:
                    old = float(el.text)
                    el.text = repr(old * scale)
                    changed.append((key, old, old * scale))

    if not changed:
        raise SystemExit(f"ERROR: no base_link inertial block found in {src_sdf}")

    tree.write(dst_sdf, encoding="UTF-8", xml_declaration=True)
    print(f"Wrote mass variant (x{scale}) to {dst_sdf}")
    for key, old, new in changed:
        print(f"  {key}: {old} -> {new}")
    print(
        "\nRelaunch with:\n"
        f"  export GZ_SIM_RESOURCE_PATH={os.path.dirname(dst_dir)}:$GZ_SIM_RESOURCE_PATH"
    )


def run_case(env, controller, kind, sp_frd, cfg):
    """One 30 s evaluation under one disturbance case."""
    sp_flu = frd_to_flu(sp_frd)
    obs, _ = env.reset(options={"eval_setpoint": sp_flu})

    if controller["type"] == "pid":
        controller["pid"].reset(measurement=flu_to_frd(obs[3:6]))

    n_steps = int(cfg["eval_time"] / cfg["dt"])
    t_hist, actual_hist, desired_hist, action_hist = [], [], [], []
    # Episode index per recorded step. A 30 s case contains ~10 separate step
    # responses; without this the metrics are computed over all of them
    # concatenated, so rise/settling describe only the first episode and
    # overshoot is the worst single value across all of them.
    ep_hist, ep = [], 0

    if kind == "wind":
        clear_force(cfg["topic"])
        apply_force(cfg["topic"], cfg["wind_force"])
    elif kind == "gust":
        clear_force(cfg["topic"])

    gust_on = False
    for step in range(n_steps):
        if kind == "gust":
            phase = step % cfg["gust_every"]
            want = phase < cfg["gust_steps"]
            if want and not gust_on:
                apply_force(cfg["topic"], cfg["gust_force"])
                gust_on = True
            elif not want and gust_on:
                clear_force(cfg["topic"])
                gust_on = False

        if controller["type"] == "ppo":
            action, _ = controller["model"].predict(obs, deterministic=True)
        else:
            meas_frd = flu_to_frd(obs[3:6])
            err_frd = sp_frd - meas_frd
            u_sat, _ = controller["pid"](err_frd, meas_frd, cfg["dt"])
            # evaluate_pid.py passes u_sat straight to env.step with no frame
            # conversion. frd_to_flu negates y and z, which inverts pitch and
            # yaw feedback; the published gains are tuned against the
            # unconverted path, so match it exactly.
            action = u_sat

        obs_next, _, terminated, truncated, _ = env.step(action)
        meas_frd = flu_to_frd(obs_next[3:6])

        ep_hist.append(ep)
        t_hist.append(step * cfg["dt"])
        desired_hist.append(sp_frd.copy())
        actual_hist.append(meas_frd.copy())
        action_hist.append(np.asarray(action, dtype=np.float32).copy())

        obs = obs_next
        if terminated or truncated:
            obs, _ = env.reset(options={"eval_setpoint": sp_flu})
            if controller["type"] == "pid":
                controller["pid"].reset(measurement=flu_to_frd(obs[3:6]))
            # The world reset may drop the persistent wrench. Re-arm it.
            if kind == "wind":
                clear_force(cfg["topic"])
                apply_force(cfg["topic"], cfg["wind_force"])
            elif kind == "gust":
                clear_force(cfg["topic"])
                gust_on = False
            ep += 1

    if kind in ("wind", "gust"):
        clear_force(cfg["topic"])

    return (
        np.array(t_hist),
        np.array(actual_hist),
        np.array(desired_hist),
        np.array(action_hist),
        np.array(ep_hist),
    )


def main(args=None):
    mass_scale = _take_flag("--emit-mass-variant", 0.0, float)
    if mass_scale > 0:
        src = _take_flag(
            "--src-sdf",
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "resources",
                "autorl_drone",
                "x500_base",
                "model.sdf",
            ),
            str,
        )
        dst = _take_flag("--variant-dir", f"./x500_base_m{mass_scale:g}", str)
        emit_mass_variant(mass_scale, os.path.abspath(src), os.path.abspath(dst))
        return

    which = _take_flag("--controller", "ppo", str)
    case = _take_flag("--case", "all", str)
    dt = _take_flag("--dt", 0.004, float)
    eval_time = _take_flag("--eval-time", 30.0, float)
    force = _take_flag("--force", 2.0, float)
    offset_z = _take_flag("--offset-z", 0.0, float)
    globals()["FORCE_OFFSET"] = (0.0, 0.0, float(offset_z))
    print(f"[disturbance] force_offset (link frame): {FORCE_OFFSET}")
    gust_force = _take_flag("--gust-force", 8.0, float)
    gust_every = _take_flag("--gust-every", 250, int)
    gust_steps = _take_flag("--gust-steps", 25, int)
    out_csv = _take_flag("--out", f"metrics_disturbance_{which}.csv", str)
    model_path_cli = _take_flag("--model", "", str)

    cases = ["none", "wind", "gust"] if case == "all" else [case]

    rclpy.init(args=args)
    env = gym.make("Autopilot-RL-v0")
    base = env  # was env.unwrapped -- the 400-step TimeLimit is part of the
                # published evaluation protocol; removing it changes the result

    controller = {"type": which}
    if which == "ppo":
        model_path = model_path_cli
        if not model_path:
            from rcl_interfaces.msg import ParameterDescriptor

            tmp = rclpy.create_node("disturbance_param_loader")
            desc = ParameterDescriptor(dynamic_typing=True)
            model_path = tmp.declare_parameter("model_path", None, desc).value
            tmp.destroy_node()
        if not model_path:
            raise RuntimeError("ERROR: pass model_path:= or --model for PPO")
        controller["model"] = PPO.load(model_path, device="cpu")
    else:
        # Gains MUST match evaluate_pid.main() -- this is the paper's baseline.
        controller["pid"] = PIDRateController(
            kp=(0.11, 0.14, 0.30),
            ki=(0.18, 0.12, 0.15),
            kd=(0.015, 0.006, 0.00),
            u_min=-1.0,
            u_max=1.0,
            integ_limit=(0.5, 0.4, 0.35),
        )

    cfg = {
        "dt": dt,
        "eval_time": eval_time,
        "wind_force": (force, 0.0, 0.0),
        "gust_force": (gust_force, gust_force * 0.5, 0.0),
        "gust_every": gust_every,
        "gust_steps": gust_steps,
        "topic": None,
    }
    if any(c in ("wind", "gust") for c in cases):
        cfg["topic"] = find_wrench_topic()
        print(f"[disturbance] wrench topic: {cfg['topic']}")

    setpoints = [
        np.array([-0.5, 0.3, -0.4], dtype=np.float32),
        np.array([0.1, -0.2, 0.6], dtype=np.float32),
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        np.array([-0.3, -0.1, 0.2], dtype=np.float32),
    ]

    rows = []
    try:
        for kind in cases:
            for i, sp in enumerate(setpoints, start=1):
                print(f"\n=== {which.upper()} | case={kind} | setpoint {i} {sp} ===")
                t, act, des, u, ep = run_case(base, controller, kind, sp, cfg)
                np.savez(
                    f"{os.path.splitext(out_csv)[0]}_raw_{kind}_sp{i}.npz",
                    t=t, actual=act, desired=des, action=u, episode=ep,
                    setpoint=sp, case=kind, controller=which,
                )
                eps = [e for e in np.unique(ep) if (ep == e).sum() >= 25]
                print(f"    episodes: {len(np.unique(ep))} total, "
                      f"{len(eps)} with >=25 steps")
                for ax in range(3):
                    # Per-episode metrics, then averaged. Each episode's time
                    # axis is re-zeroed so rise/settling/ITAE are measured from
                    # that episode's own start.
                    per = []
                    for e in eps:
                        sel = ep == e
                        te = t[sel] - t[sel][0]
                        per.append(compute_axis_metrics(
                            te, act[sel, ax], sp[ax], u[sel, ax]))
                    if not per:
                        continue
                    fields = ["rise_time", "overshoot_pct", "settling_time",
                              "ss_error_abs", "IAE", "ISE", "ITAE", "effort_L1",
                              "effort_L2", "sat_frac", "time_in_thresh_frac"]
                    avg = {f: float(np.nanmean([getattr(p, f) for p in per]))
                           for f in fields}
                    m = type("M", (), avg)()
                    m.n_episodes = len(per)
                    rows.append(
                        {
                            "controller": which,
                            "case": kind,
                            "setpoint_index": i,
                            "axis": ["Roll_X", "Pitch_Y", "Yaw_Z"][ax],
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
        df.to_csv(out_csv, index=False)
        print("\n=== per-case means ===")
        print(
            df.groupby("case")[
                ["rise_time_s", "overshoot_pct", "settling_time_s", "ss_error_rad_s"]
            ]
            .mean()
            .to_string()
        )
        print(f"\nWritten to {os.path.abspath(out_csv)}")

    finally:
        if hasattr(base, "main_node"):
            base.main_node.destroy_node()
        env.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
