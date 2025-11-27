#!/usr/bin/env python3
import os, zipfile, io, json, pickle, numpy as np, argparse


def load_sb3_data(zip_path):
    """Load SB3 .zip model and return its data dict (JSON or pickle)."""
    with zipfile.ZipFile(zip_path, "r") as z:
        name = [n for n in z.namelist() if n.endswith("/data") or n == "data"]
        if not name:
            raise FileNotFoundError("No 'data' file found inside zip.")
        with z.open(name[0]) as f:
            raw = f.read()
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return pickle.loads(raw)


def safe_val(v):
    """Convert value to printable clean text."""
    if callable(v):
        return "function (dynamic)"
    if isinstance(v, dict) and ":type:" in v:
        return "function (dynamic)"
    if isinstance(v, (list, np.ndarray)) and len(v) > 0:
        return f"array(len={len(v)})"
    return v


def summarize_dict(d, f, indent=0):
    """Recursively print nested dicts and arrays."""
    sp = "  " * indent
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, dict):
                f.write(f"{sp}{k}:\n")
                summarize_dict(v, f, indent + 1)
            elif isinstance(v, (list, np.ndarray)):
                f.write(f"{sp}{k}: array(shape={len(v)}) sample={np.round(v[:10],4)}\n")
            else:
                f.write(f"{sp}{k}: {safe_val(v)}\n")
    else:
        f.write(f"{sp}{safe_val(d)}\n")


def _parse_numeric_array(val):
    """Robust parser: handles lists, ndarrays, or stringified numeric arrays."""
    if isinstance(val, (list, np.ndarray)):
        return np.array(val, dtype=float)
    if isinstance(val, str):
        # Clean and split the string safely
        txt = val.strip().replace("[", "").replace("]", "")
        txt = txt.replace("\n", " ")
        parts = [p for p in txt.split(" ") if p not in ("",)]
        arr = []
        for p in parts:
            try:
                if p.lower() in ("-inf", "inf", "+inf"):
                    arr.append(np.inf if p[0] != "-" else -np.inf)
                else:
                    arr.append(float(p))
            except ValueError:
                continue
        return np.array(arr, dtype=float)
    return np.array([], dtype=float)


def write_training_info(data, out_path):
    """Write detailed PPO training metadata and normalization info."""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("### PPO Training Metadata Export ###\n\n")

        #  Summary Table
        f.write(" Parameters Required for Online Tuning \n")
        f.write(
            "+----------------------+-----------------------------+------------------+\n"
        )
        f.write(
            "| Parameter            | Description                 | Value            |\n"
        )
        f.write(
            "+----------------------+-----------------------------+------------------+\n"
        )

        def val(k):
            if k in data:
                return safe_val(data[k])
            return "—"

        obs_mean = obs_var = "—"
        if "obs_rms" in data and isinstance(data["obs_rms"], dict):
            mean_arr = np.array(data["obs_rms"].get("mean", []))
            var_arr = np.array(data["obs_rms"].get("var", []))
            if mean_arr.size > 0:
                obs_mean = f"{np.mean(mean_arr):.6f} (avg of {mean_arr.size})"
            if var_arr.size > 0:
                obs_var = f"{np.mean(var_arr):.6f} (avg of {var_arr.size})"

        table = [
            ("obs_rms.mean", "Observation normalization mean", obs_mean),
            ("obs_rms.var", "Observation normalization variance", obs_var),
            ("gamma", "Discount factor for future rewards", val("gamma")),
            ("learning_rate", "Step size for policy update", val("learning_rate")),
            ("gae_lambda", "GAE tradeoff between bias/variance", val("gae_lambda")),
            ("clip_range", "Policy update clipping range", val("clip_range")),
            ("ent_coef", "Entropy coefficient", val("ent_coef")),
            ("vf_coef", "Value loss coefficient", val("vf_coef")),
            ("max_grad_norm", "Gradient clipping norm", val("max_grad_norm")),
            ("n_steps", "Steps per PPO rollout", val("n_steps")),
            ("batch_size", "Mini-batch size per epoch", val("batch_size")),
            ("n_epochs", "Optimization epochs per update", val("n_epochs")),
        ]

        for p, desc, v in table:
            f.write(f"| {p:<20} | {desc:<27} | {str(v):<16} |\n")

        f.write(
            "+----------------------+-----------------------------+------------------+\n\n"
        )

        #  Core PPO Hyperparameters
        f.write(" Core PPO Hyperparameters \n")
        for key in [
            "n_steps",
            "batch_size",
            "n_epochs",
            "gamma",
            "learning_rate",
            "gae_lambda",
            "clip_range",
            "ent_coef",
            "vf_coef",
            "max_grad_norm",
        ]:
            if key in data:
                f.write(f"{key}: {safe_val(data[key])}\n")
        f.write("\n")

        # Observation Normalization
        f.write(" Observation Normalization (obs_rms) \n")
        if "obs_rms" in data:
            v = data["obs_rms"]
            if isinstance(v, dict):
                mean = np.array(v.get("mean", []), dtype=float)
                var = np.array(v.get("var", []), dtype=float)
                count = v.get("count")
                f.write(f"len(mean)={len(mean)} len(var)={len(var)} count={count}\n")
                if mean.size > 0:
                    f.write("mean_full: " + " ".join(f"{x:.6f}" for x in mean) + "\n")
                if var.size > 0:
                    f.write("var_full: " + " ".join(f"{x:.6f}" for x in var) + "\n")
                if count is not None:
                    f.write(f"count: {count}\n")
        if "clip_obs" in data:
            f.write(f"clip_obs: {safe_val(data['clip_obs'])}\n")
        f.write("\n")

        # Observation and Action Space Bounds
        f.write(" Spaces (shapes and bounds) \n")
        obs = data.get("observation_space", {})
        act = data.get("action_space", {})

        def dump_space(name, s):
            low = _parse_numeric_array(s.get("low", []))
            high = _parse_numeric_array(s.get("high", []))
            shp = s.get("_shape", [])
            f.write(f"{name}.shape: {shp}\n")
            if low.size > 0:
                low_txt = " ".join(
                    f"{x:.6f}" if np.isfinite(x) else "-inf" for x in low
                )
                f.write(f"{name}.low : {low_txt}\n")
            if high.size > 0:
                high_txt = " ".join(
                    f"{x:.6f}" if np.isfinite(x) else "inf" for x in high
                )
                f.write(f"{name}.high: {high_txt}\n")

        dump_space("observation_space", obs)
        dump_space("action_space", act)
        f.write("\n")

        f.write(" Additional Training Data \n")
        for key in [
            "episode_reward",
            "ep_info_buffer",
            "ep_len_buffer",
            "total_timesteps",
            "num_timesteps",
            "timesteps_since_restore",
        ]:
            if key in data:
                f.write(f"{key}: {safe_val(data[key])}\n")
        f.write("\n")

        #  All Remaining Keys (raw)
        f.write(" All Remaining Keys (raw) \n")
        for k, v in data.items():
            if k not in [
                "obs_rms",
                "clip_obs",
                "episode_reward",
                "ep_info_buffer",
                "ep_len_buffer",
                "total_timesteps",
                "num_timesteps",
                "timesteps_since_restore",
                "n_steps",
                "batch_size",
                "n_epochs",
                "gamma",
                "learning_rate",
                "gae_lambda",
                "clip_range",
                "ent_coef",
                "vf_coef",
                "max_grad_norm",
            ]:
                f.write(f"{k}:\n")
                summarize_dict(v, f, indent=1)
                f.write("\n")

    print(f"Training info written to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to PPO model .zip file")
    args = parser.parse_args()

    ppo_zip_path = args.model

    data = load_sb3_data(ppo_zip_path)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, "ppo_training_info.txt")
    write_training_info(data, out_path)
