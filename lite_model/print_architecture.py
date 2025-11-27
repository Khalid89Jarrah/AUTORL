#!/usr/bin/env python3
import os, zipfile, io, json, pickle, torch, numpy as np, argparse


def load_sb3_data(zip_path):
    """Load SB3 .zip model metadata."""
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


def extract_policy_state(zip_path):
    """Extract and load policy.pth as a state_dict."""
    with zipfile.ZipFile(zip_path, "r") as z:
        policy_file = [n for n in z.namelist() if n.endswith("policy.pth")]
        if not policy_file:
            raise FileNotFoundError("policy.pth not found inside zip.")
        with z.open(policy_file[0]) as f:
            buffer = io.BytesIO(f.read())
    return torch.load(buffer, map_location="cpu", weights_only=False)


def print_network_structure(meta):
    """Print human-readable PPO network architecture."""
    print("\n=== PPO Policy Architecture ===\n")

    obs_shape = meta.get("observation_space", {}).get("_shape", ["?"])
    act_shape = meta.get("action_space", {}).get("_shape", ["?"])
    obs_dim = obs_shape[0] if obs_shape else "?"
    act_dim = act_shape[0] if act_shape else "?"
    hidden = [64, 64]
    if "policy_kwargs" in meta and meta["policy_kwargs"]:
        hidden = meta["policy_kwargs"].get("net_arch", hidden)

    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension:      {act_dim}")
    print(f"Hidden layers:         {hidden}")
    print("Activation:            tanh (default for MlpPolicy)\n")

    # ==== Action space bounds ====
    act_low = meta.get("action_space", {}).get("low", [])
    act_high = meta.get("action_space", {}).get("high", [])
    if isinstance(act_low, str):
        act_low = act_low.replace("[", "").replace("]", "").split()
    if isinstance(act_high, str):
        act_high = act_high.replace("[", "").replace("]", "").split()

    if act_low and act_high:
        print(f"Action bounds:         low={act_low[:5]} ... high={act_high[:5]} ...\n")

    print("Network structure:")
    print(" Input (15)")
    print("   │")
    print(" ├─► Dense(64) ── tanh")
    print(" ├─► Dense(64) ── tanh")
    print("   │")
    print(" ├─► Actor head:  Dense(3) ── tanh  →  action(3)")
    print(" └─► Critic head: Dense(1) ── linear →  value(1)\n")


def write_all_layers(state_dict, meta, out_path):
    """Write full model parameters and architecture to text."""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("### PPO Full Model Export ###\n\n")
        f.write("=== PPO Policy Architecture ===\n")
        obs_dim = meta.get("observation_space", {}).get("_shape", [15])[0]
        act_dim = meta.get("action_space", {}).get("_shape", [3])[0]
        f.write(f"Observation dimension: {obs_dim}\n")
        f.write(f"Action dimension:      {act_dim}\n")
        f.write("Hidden layers:         [64, 64]\n")
        f.write("Activation:            tanh (default for MlpPolicy)\n\n")

        # ==== Add action bounds to file ====
        act_low = meta.get("action_space", {}).get("low", [])
        act_high = meta.get("action_space", {}).get("high", [])
        if isinstance(act_low, str):
            act_low = act_low.replace("[", "").replace("]", "").split()
        if isinstance(act_high, str):
            act_high = act_high.replace("[", "").replace("]", "").split()
        if act_low and act_high:
            f.write("=== Action Space Bounds ===\n")
            f.write("low:  " + " ".join(act_low) + "\n")
            f.write("high: " + " ".join(act_high) + "\n\n")

        f.write("Network structure:\n")
        f.write(" Input (15)\n")
        f.write("   │\n")
        f.write(" ├─► Dense(64) ── tanh\n")
        f.write(" ├─► Dense(64) ── tanh\n")
        f.write("   │\n")
        f.write(" ├─► Actor head:  Dense(3) ── tanh  →  action(3)\n")
        f.write(" └─► Critic head: Dense(1) ── linear →  value(1)\n\n")

        # ==== ACTOR ====
        f.write("\n============================\n")
        f.write("=== POLICY (ACTOR) NETWORK ===\n")
        f.write("============================\n")
        for key, tensor in state_dict.items():
            if "policy_net" in key or "action_net" in key:
                arr = tensor.numpy()
                f.write(f"\n# {key}\nshape={arr.shape}\n")
                if "policy_net.0" in key:
                    f.write("# First hidden layer (64) from input(15)\n")
                elif "policy_net.2" in key:
                    f.write("# Second hidden layer (64) from previous(64)\n")
                elif "action_net" in key:
                    f.write("# Actor head: outputs 3 actions (roll, pitch, yaw)\n")
                if arr.ndim == 1:
                    for i, val in enumerate(arr):
                        f.write(f"b{i+1}={val:.6f}\n")
                elif arr.ndim == 2:
                    for i, row in enumerate(arr):
                        f.write(f"row {i:03d}: ")
                        f.write(" ".join(f"{x:.6f}" for x in row))
                        f.write("\n")

        # ==== CRITIC ====
        f.write("\n============================\n")
        f.write("=== CRITIC (VALUE) NETWORK ===\n")
        f.write("============================\n")
        for key, tensor in state_dict.items():
            if "value_net" in key:
                arr = tensor.numpy()
                f.write(f"\n# {key}\nshape={arr.shape}\n")
                if "value_net.0" in key:
                    f.write("# First hidden layer (64) from input(15)\n")
                elif "value_net.2" in key:
                    f.write("# Second hidden layer (64) from previous(64)\n")
                elif "value_net.weight" in key and "mlp_extractor" not in key:
                    f.write("# Final critic output (1 value estimate)\n")
                if arr.ndim == 1:
                    for i, val in enumerate(arr):
                        f.write(f"b{i+1}={val:.6f}\n")
                elif arr.ndim == 2:
                    for i, row in enumerate(arr):
                        f.write(f"row {i:03d}: ")
                        f.write(" ".join(f"{x:.6f}" for x in row))
                        f.write("\n")

        # ==== LOG_STD ====
        f.write("\n============================\n")
        f.write("=== LOG_STD (Policy Noise) ===\n")
        f.write("============================\n")
        for key, tensor in state_dict.items():
            if "log_std" in key:
                arr = tensor.numpy()
                f.write(f"\n# {key}\nshape={arr.shape}\n")
                f.write("# Log standard deviation for stochastic policy outputs\n")
                for i, val in enumerate(arr):
                    f.write(f"b{i+1}={val:.6f}\n")

        f.write("\n\n### END OF FILE ###\n")

    print(f"All weights and biases written to: {out_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to PPO model .zip file")
    args = parser.parse_args()

    ppo_zip_path = args.model

    meta = load_sb3_data(ppo_zip_path)
    state_dict = extract_policy_state(ppo_zip_path)
    print_network_structure(meta)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, "ppo_weights_full.txt")
    write_all_layers(state_dict, meta, out_path)
