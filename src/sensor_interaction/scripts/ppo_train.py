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

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import set_random_seed

from sensor_interaction.autorl_world import Auto_RL, ABLATABLE_TERMS
from sensor_interaction.MainNode import MainNode

# ---------------------------------------------------------------------------
# Campaign A / B configuration
# ---------------------------------------------------------------------------
# EVAL_INTERVAL_EPISODES * max_steps timesteps are collected between each
# checkpoint + evaluation. With max_steps = 400 this is 100_000 timesteps.
EVAL_INTERVAL_EPISODES = 250

# 0 = fresh start, or e.g. 400000 to continue from that timestep checkpoint.
# NOTE: this is counted in TIMESTEPS, not episodes.
START_TRAINING_FROM = 0

MODEL_BASENAME = "ppo_model_ref"

# Campaign B writes to a different directory than Campaign A so the two sets
# of artefacts never mix. Override with AUTORL_MODEL_DIR / AUTORL_LOG_DIR /
# AUTORL_TB_DIR (this is how runB.sbatch keeps each arm's outputs separate).
MODEL_DIR = os.environ.get("AUTORL_MODEL_DIR", "/opt/autorl_ws/models")
LOG_DIR = os.environ.get("AUTORL_LOG_DIR", "./training_logs")
TB_DIR = os.environ.get("AUTORL_TB_DIR", "./ppo_tensorboard")

# PPO hyperparameters (unchanged from the published configuration)
GAMMA = 0.99
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
N_EPOCHS = 10
HORIZON = 2048


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
        (history["reward"], "Average Reward", f"Training Progress - PPO{title_suffix}"),
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


def train_model(
    env,
    total_timesteps,
    max_steps,
    seed,
    run_tag,
    arm,
    gamma=GAMMA,
    learning_rate=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    n_epochs=N_EPOCHS,
    horizon=HORIZON,
):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # FIX (Campaign A): this path was previously a plain string containing the
    # literal text "{MODEL_BASENAME}", not an f-string, so the final model was
    # written to a file actually named "{MODEL_BASENAME}.zip".
    model_save_path = f"{MODEL_DIR}/{run_tag}.zip"
    csv_path = f"{LOG_DIR}/{run_tag}_progress.csv"
    png_path = f"{LOG_DIR}/{run_tag}_progress.png"

    if START_TRAINING_FROM == 0:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            gamma=gamma,
            batch_size=batch_size,
            n_epochs=n_epochs,
            n_steps=horizon,
            tensorboard_log=f"{TB_DIR}/{arm}/seed_{seed}/",
            verbose=1,
            device="cpu",
            seed=seed,
        )
        print(f"Starting PPO training from scratch (arm={arm}, seed={seed}).")
    else:
        model_path = f"{MODEL_DIR}/{run_tag}_{START_TRAINING_FROM}.zip"
        print(f"Continuing training from checkpoint: {model_path}")
        model = PPO.load(model_path, env=env, seed=seed)

    save_interval = EVAL_INTERVAL_EPISODES * max_steps

    # Training metrics. These lists were previously created and never appended
    # to, so every plot produced by this script was empty.
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
        _save_progress_plot(png_path, history, title_suffix=f"  [{arm}, seed {seed}]")

        model_filename = f"{MODEL_DIR}/{run_tag}_{absolute_timesteps}.zip"
        model.save(model_filename)
        print(
            f"[{arm} seed {seed}] {absolute_timesteps}/"
            f"{START_TRAINING_FROM + total_timesteps} timesteps - "
            f"checkpoint saved: {model_filename}"
        )

    model.save(model_save_path)
    print(f"Final model saved as: {model_save_path}")
    print(f"Progress CSV: {csv_path}")
    print(f"Progress plot: {png_path}")


def evaluate_model(env, model, num_episodes=5):
    """
    Evaluates the PPO model and returns multiple training metrics.
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
        f"Evaluation: Reward={avg_reward:.2f}, Action Mag={avg_action_magnitude:.2f}, Tracking Error={avg_angular_velocity_error:.2f}, Ep Length={avg_episode_length:.2f}"
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
    arm = _take_flag("--ablate", "AUTORL_ABLATE", "full").strip().lower()

    if total_timesteps <= 0:
        print(
            "ERROR: no training budget given.\n"
            "  Set it explicitly, e.g.\n"
            "    ros2 launch sensor_interaction node_launch.py algorithm:=ppo "
            "gui:=false mode:=training seed:=0 timesteps:=1200000 ablate:=full\n"
            "  or export AUTORL_TIMESTEPS=1200000"
        )
        sys.exit(1)

    if arm not in (["full", "none", ""] + ABLATABLE_TERMS):
        print(
            f"ERROR: unknown reward ablation '{arm}'.\n"
            f"  Valid values: full, {', '.join(ABLATABLE_TERMS)}"
        )
        sys.exit(1)
    if arm in ("none", ""):
        arm = "full"

    run_tag = f"{MODEL_BASENAME}_{arm}_seed{seed}"

    rclpy.init(args=args)
    # Campaign B: `ablate` is forwarded through create_auto_rl_env to Auto_RL.
    env = gym.make("Autopilot-RL-v0", ablate=arm)
    check_env(env)

    # Seed everything: python/numpy/torch via SB3, the env RNG that samples the
    # angular-velocity setpoint, and the action space sampler.
    set_random_seed(seed)
    env.reset(seed=seed)
    env.action_space.seed(seed)

    logger = rclpy.logging.get_logger("my_logger")
    logger.info(
        f"Starting PPO training (arm={arm}, seed={seed}, timesteps={total_timesteps})"
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
            arm=arm,
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