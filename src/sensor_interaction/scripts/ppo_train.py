#!/usr/bin/env python3
import os
import sys
import time
import csv

# Headless: the training node runs under `ros2 launch` with no display, so the
# Agg backend must be selected BEFORE pyplot is imported. (The evaluation
# scripts already do this; this one previously did not and called plt.show().)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import psutil
import rclpy
import gymnasium as gym

from stable_baselines3 import PPO, TD3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.utils import set_random_seed

from sensor_interaction.autorl_world import Auto_RL
from sensor_interaction.MainNode import MainNode

# ---------------------------------------------------------------------------
# Campaign A / C configuration
# ---------------------------------------------------------------------------
# EVAL_INTERVAL_EPISODES * max_steps timesteps are collected between each
# checkpoint + evaluation. With max_steps = 750 (Campaign A fix: matches the
# real 0.004 s physics step to the environment's 3.0 s internal truncation)
# this is 187_500 timesteps between checkpoints.
EVAL_INTERVAL_EPISODES = 250

# 0 = fresh start, or e.g. 400000 to continue from that timestep checkpoint.
# NOTE: this is counted in TIMESTEPS, not episodes.
START_TRAINING_FROM = 0

MODEL_DIR = "/opt/autorl_ws/models"
LOG_DIR = "./training_logs"
TB_DIR = "./ppo_tensorboard"

# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------
# The Auto_RL environment (autorl_world.py) is NOT touched by this file.
# Only the learning algorithm driving it changes - that is the claim under
# test (R1-2: "framework is algorithm-independent").
#
# Both algorithms use the SAME network size ([64, 64]) as the published PPO
# policy, so the comparison is not confounded by capacity. SB3's TD3 default
# is [400, 300], which would be a much larger network than the
# 10,628-parameter policy reported in the paper.
NET_ARCH = [64, 64]

# PPO hyperparameters: UNCHANGED from the published configuration. This path
# is the paper's baseline result and is not touched by adding TD3.
PPO_KWARGS = dict(
    learning_rate=1e-4,
    gamma=0.99,
    batch_size=64,
    n_epochs=10,
    n_steps=2048,
)

# TD3 hyperparameters.
#
# CORRECTED from the first Campaign C attempt, which used the SB3 defaults
# (learning_rate=1e-3, target_policy_noise=0.2, target_noise_clip=0.5) and
# collapsed: the actor's action magnitude drifted from 0.033 to 1.372 over
# 2M steps (checkpoints evaluated on the fixed 31-setpoint bank: ss_error
# 0.40-0.57 rad/s, sat_frac 9-14%, vs PPO's ss_error ~0.015, sat_frac 0.00%).
# That shape - fast early improvement, then monotonic drift to saturation -
# is the standard TD3 Q-overestimation failure mode for a deterministic
# actor. Two changes address this directly:
#
#   1. learning_rate: 1e-3 -> 3e-4. 1e-3 is aggressive for a deterministic
#      actor with no entropy regularization (unlike SAC). 3e-4 is SB3's own
#      default for SAC on continuous control and a more conservative choice
#      here.
#   2. target_policy_noise / target_noise_clip made EXPLICIT rather than
#      left at SB3 defaults (0.2 / 0.5). Those defaults are large relative
#      to this environment's [-1, 1] action range. 0.1 / 0.3 is used here.
#
# These are hypotheses consistent with the observed collapse, not a proven
# fix - the corrected config has not yet been run. Re-evaluate on the fixed
# 31-setpoint bank after training and compare against PID/PPO before
# drawing conclusions.
TD3_KWARGS = dict(
    learning_rate=3e-4,  # was 1e-3
    gamma=0.99,
    batch_size=256,
    buffer_size=1_000_000,
    learning_starts=10_000,
    train_freq=1,
    gradient_steps=1,
    tau=0.005,
    policy_delay=2,
    target_policy_noise=0.1,  # was SB3 default 0.2 (unset previously)
    target_noise_clip=0.3,  # was SB3 default 0.5 (unset previously)
)

ALGOS = ("ppo", "td3")


def build_model(algo, env, seed, tb_path):
    """Construct the requested SB3 algorithm on the AutoRL environment."""
    common = dict(
        policy="MlpPolicy",
        env=env,
        tensorboard_log=tb_path,
        verbose=1,
        device="cpu",
        seed=seed,
    )

    if algo == "ppo":
        return PPO(
            **common,
            policy_kwargs=dict(net_arch=dict(pi=NET_ARCH, vf=NET_ARCH)),
            **PPO_KWARGS,
        )

    if algo == "td3":
        # TD3's actor is deterministic, so exploration relies on action
        # noise. sigma=0.1 on a [-1, 1] action space is the standard choice.
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
        )
        return TD3(
            **common,
            policy_kwargs=dict(net_arch=NET_ARCH),
            action_noise=action_noise,
            **TD3_KWARGS,
        )

    raise ValueError(f"Unknown algorithm '{algo}'")


def load_model(algo, path, env, seed):
    cls = {"ppo": PPO, "td3": TD3}[algo]
    return cls.load(path, env=env, seed=seed)


# ---------------------------------------------------------------------------
# Command line / environment configuration
# ---------------------------------------------------------------------------
def _take_flag(flag, env_var, default):
    """Read a value from `--flag VALUE` (removing it from sys.argv so rclpy
    never sees it), falling back to an environment variable, then `default`."""
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            value = sys.argv[i + 1]
            del sys.argv[i : i + 2]
            return value
    return os.environ.get(env_var, default)


def _take_int_flag(flag, env_var, default):
    return int(_take_flag(flag, env_var, str(default)))


# Get the PIDs of Gazebo and ROS-related bridges
def get_gazebo_pids():
    """Get the PIDs of relevant Gazebo processes."""
    pids = []
    for proc in psutil.process_iter(attrs=["pid", "name"]):
        if "gzserver" in proc.info["name"] or "parameter_bridge" in proc.info["name"]:
            pids.append(proc.info["pid"])
    return pids


# Terminate the node and related processes
def close(main_node):
    """Gracefully closes the node and terminates related processes."""
    try:
        print("Destroy node")
        main_node.stop_executor()
        main_node.destroy_node()
        print("Node destroyed successfully.")
    except Exception as e:
        print(f"Error during node destruction: {e}")
    finally:
        try:
            # Close the CSV file if it's open
            if hasattr(main_node, "log_file") and main_node.log_file:
                main_node.log_file.close()
                print("CSV log file closed successfully.")
        except Exception as e:
            print(f"Error closing the CSV log file: {e}")
        finally:
            try:
                rclpy.shutdown()
                print("rclpy shutdown complete.")
            except Exception as e:
                print(f"Error during rclpy shutdown: {e}")
            finally:
                pids = get_gazebo_pids()
                for pid in pids:
                    try:
                        os.kill(pid, 9)  # Forcefully terminate
                    except Exception as e:
                        print(f"Error terminating process {pid}: {e}")
                sys.exit(0)


def _write_progress_csv(csv_path, history):
    """Rewrite the full progress CSV after every evaluation, so a run that is
    killed part-way still leaves a complete, readable learning curve on disk."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timesteps",
                "avg_reward",
                "avg_action_magnitude",
                "avg_angular_velocity_error",
                "avg_episode_length",
            ]
        )
        for row in zip(
            history["timesteps"],
            history["reward"],
            history["action_magnitude"],
            history["angular_velocity_error"],
            history["episode_length"],
        ):
            writer.writerow(row)


def _save_progress_plot(png_path, history, title_suffix=""):
    """Static learning-curve figure (no interactive display: headless run)."""
    if not history["timesteps"]:
        return

    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    series = [
        (history["reward"], "Average Reward", f"Training Progress{title_suffix}"),
        (history["action_magnitude"], "Mean Action Magnitude", "Action Magnitude"),
        (
            history["angular_velocity_error"],
            "Mean Tracking Error",
            "Tracking Error (Angular Velocity)",
        ),
        (history["episode_length"], "Steps per Episode", "Episode Lengths"),
    ]
    for ax, (data, ylabel, title) in zip(axs, series):
        ax.plot(history["timesteps"], data, marker="o", linestyle="-")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axs[-1].set_xlabel("Timesteps")

    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def train_model(env, total_timesteps, max_steps, seed, run_tag, algo):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # FIX (Campaign A): this path was previously a plain string containing
    # the literal text "{MODEL_BASENAME}", not an f-string, so the final
    # model was written to a file actually named "{MODEL_BASENAME}.zip".
    model_save_path = f"{MODEL_DIR}/{run_tag}.zip"
    csv_path = f"{LOG_DIR}/{run_tag}_progress.csv"
    png_path = f"{LOG_DIR}/{run_tag}_progress.png"

    if START_TRAINING_FROM == 0:
        model = build_model(algo, env, seed, f"{TB_DIR}/{algo}/seed_{seed}/")
        print(f"Starting {algo.upper()} training from scratch (seed={seed}).")
    else:
        model_path = f"{MODEL_DIR}/{run_tag}_{START_TRAINING_FROM}.zip"
        print(f"Continuing training from checkpoint: {model_path}")
        model = load_model(algo, model_path, env, seed)

    # Report the trainable parameter count: the paper claims a compact
    # policy (10,628 parameters for PPO). This makes the comparison
    # auditable against the same network size for TD3.
    n_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    print(f"[{algo.upper()}] trainable policy parameters: {n_params}")

    save_interval = EVAL_INTERVAL_EPISODES * max_steps

    # Training metrics. These lists were previously created and never
    # appended to, so every plot produced by this script was empty.
    history = {
        "timesteps": [],
        "reward": [],
        "action_magnitude": [],
        "angular_velocity_error": [],
        "episode_length": [],
    }

    elapsed = 0
    while elapsed < total_timesteps:
        chunk = min(save_interval, total_timesteps - elapsed)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        elapsed += chunk
        absolute_timesteps = START_TRAINING_FROM + elapsed

        # Evaluate the current policy
        (
            avg_reward,
            avg_action_magnitude,
            avg_angular_velocity_error,
            avg_episode_length,
        ) = evaluate_model(env, model, num_episodes=10)

        history["timesteps"].append(absolute_timesteps)
        history["reward"].append(float(avg_reward))
        history["action_magnitude"].append(float(avg_action_magnitude))
        history["angular_velocity_error"].append(float(avg_angular_velocity_error))
        history["episode_length"].append(float(avg_episode_length))

        model.logger.record("eval/avg_reward", float(avg_reward))
        model.logger.record("eval/action_magnitude", float(avg_action_magnitude))
        model.logger.record(
            "eval/angular_velocity_error", float(avg_angular_velocity_error)
        )
        model.logger.record("eval/episode_length", float(avg_episode_length))
        model.logger.dump(step=absolute_timesteps)

        _write_progress_csv(csv_path, history)
        _save_progress_plot(
            png_path, history, title_suffix=f" - {algo.upper()} [seed {seed}]"
        )

        model_filename = f"{MODEL_DIR}/{run_tag}_{absolute_timesteps}.zip"
        model.save(model_filename)
        print(
            f"[{algo} seed {seed}] {absolute_timesteps}/"
            f"{START_TRAINING_FROM + total_timesteps} timesteps - "
            f"checkpoint saved: {model_filename}"
        )

        # WARNING (unresolved, flagged not fixed): action magnitude drifting
        # well past 1.0 late in training is the exact signature of the
        # collapse seen in the first TD3 attempt. This does not stop
        # training - it is a visible early-warning print only.
        if algo == "td3" and avg_action_magnitude > 0.9:
            print(
                f"[{algo} seed {seed}] WARNING: avg action magnitude "
                f"{avg_action_magnitude:.3f} is approaching/past the [-1,1] "
                f"bound. This matched the failure signature from the first "
                f"TD3 attempt (0.033 -> 1.372 over 2M steps). Check the "
                f"held-out evaluator, not just this in-training number."
            )

    model.save(model_save_path)
    print(f"Final model saved as: {model_save_path}")
    print(f"Progress CSV: {csv_path}")
    print(f"Progress plot: {png_path}")


def evaluate_model(env, model, num_episodes=5):
    """
    Evaluates the model and returns multiple training metrics.

    NOTE: this in-training evaluation draws episodes from the training
    distribution N(0, 0.3) via env.reset() with no eval_setpoint override,
    using only num_episodes samples. It is a rough training-time signal
    only. The first Campaign C attempt showed this can disagree sharply
    with a proper held-out evaluation (10 training-distribution episodes
    read ~0.030 rad/s error at 200k steps; the fixed 31-setpoint bank read
    0.40-0.57 rad/s at the same checkpoint). Always confirm any promising
    number here against evaluate_ppo.py's held-out setpoint bank before
    reporting it.
    """
    total_rewards = []
    total_action_magnitude = []
    total_angular_velocity_error = []
    total_episode_length = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_actions = []
        episode_errors = []
        step_count = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            episode_actions.append(np.linalg.norm(action))  # Track action magnitude
            episode_errors.append(np.linalg.norm(obs[:3]))  # Track tracking error
            done = terminated or truncated
            step_count += 1

        total_rewards.append(episode_reward)
        total_action_magnitude.append(np.mean(episode_actions))
        total_angular_velocity_error.append(np.mean(episode_errors))
        total_episode_length.append(step_count)

    avg_reward = np.mean(total_rewards)
    avg_action_magnitude = np.mean(total_action_magnitude)
    avg_angular_velocity_error = np.mean(total_angular_velocity_error)
    avg_episode_length = np.mean(total_episode_length)

    print(
        f"Evaluation: Reward={avg_reward:.2f}, Action Mag={avg_action_magnitude:.2f}, "
        f"Tracking Error={avg_angular_velocity_error:.2f}, Ep Length={avg_episode_length:.2f}"
    )
    return (
        avg_reward,
        avg_action_magnitude,
        avg_angular_velocity_error,
        avg_episode_length,
    )


def main(args=None):
    # Parsed and stripped from sys.argv before rclpy.init sees them.
    seed = _take_int_flag("--seed", "AUTORL_SEED", 0)
    total_timesteps = _take_int_flag("--timesteps", "AUTORL_TIMESTEPS", 0)
    algo = _take_flag("--algo", "AUTORL_ALGO", "ppo").strip().lower()

    if total_timesteps <= 0:
        print(
            "ERROR: no training budget given.\n"
            "  Set it explicitly, e.g.\n"
            "    ros2 launch sensor_interaction node_launch.py algorithm:=ppo "
            "gui:=false mode:=training seed:=0 timesteps:=2000000 rl_algo:=td3\n"
            "  or export AUTORL_TIMESTEPS=2000000"
        )
        sys.exit(1)

    if algo not in ALGOS:
        print(f"ERROR: unknown algorithm '{algo}'. Valid values: {', '.join(ALGOS)}")
        sys.exit(1)

    run_tag = f"{algo}_model_ref_seed{seed}"

    rclpy.init(args=args)
    env = gym.make("Autopilot-RL-v0")
    check_env(env)

    # Seed everything: python/numpy/torch via SB3, the env RNG that samples
    # the angular-velocity setpoint, and the action space sampler.
    set_random_seed(seed)
    env.reset(seed=seed)
    env.action_space.seed(seed)

    logger = rclpy.logging.get_logger("my_logger")
    logger.info(
        f"Starting {algo.upper()} training (seed={seed}, timesteps={total_timesteps})"
    )
    try:
        max_steps = env.spec.max_episode_steps
        start_time = time.time()  # Store start time
        train_model(
            env,
            total_timesteps=total_timesteps,
            max_steps=max_steps,
            seed=seed,
            run_tag=run_tag,
            algo=algo,
        )
        end_time = time.time()  # Store end time
        print(
            f"Training time: {end_time - start_time:.2f} seconds"
        )  # Print training duration

    except Exception as e:
        print(f"Error during execution: {e}")
    finally:
        if hasattr(env.unwrapped, "main_node"):
            env.unwrapped.main_node.destroy_node()
        env.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()