#!/usr/bin/env python3
import gymnasium as gym
from stable_baselines3 import PPO
from sensor_interaction.autorl_world import Auto_RL
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import rclpy
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import logging
import numpy as np
import os
from sensor_interaction.MainNode import MainNode
import subprocess
import psutil
import sys
import time
import csv

EVAL_INTERVAL_EPISODES = 1000
START_TRAINING_FROM = 0  # 0 = fresh start, or e.g. 8000 to continue training
TOTAL_EPISODES = 1000000  # total number of training episodes
MODEL_BASENAME = "ppo_model_ref"  # base name for model files


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


def train_model(
    env,
    episodes,
    max_steps,
    gamma=0.99,
    learning_rate=1e-4,
    batch_size=64,
    n_epochs=10,
    horizon=2048,
    model_save_path="/opt/autorl_ws/models/{MODEL_BASENAME}.zip",
):
    if START_TRAINING_FROM == 0:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            gamma=gamma,
            batch_size=batch_size,
            n_epochs=n_epochs,
            n_steps=horizon,
            tensorboard_log="./ppo_tensorboard/",
            verbose=1,
            device="cpu",
        )
        print("Starting PPO training from scratch.")
    else:
        model_path = f"/opt/autorl_ws/models/{MODEL_BASENAME}_{START_TRAINING_FROM}.zip"
        print(f"Continuing training from checkpoint: {model_path}")
        model = PPO.load(model_path, env=env)

    total_timesteps = max_steps * episodes

    save_interval = EVAL_INTERVAL_EPISODES * max_steps

    # Initialize lists to track training metrics
    reward_history = []
    action_magnitude_history = []
    angular_velocity_error_history = []
    episode_length_history = []
    episode_numbers = []

    # Initialize live plot
    plt.ion()
    fig, axs = plt.subplots(4, 1, figsize=(10, 12))

    axs[0].set_title("Training Progress - PPO")
    axs[0].set_ylabel("Average Reward")

    axs[1].set_title("Action Magnitude")
    axs[1].set_ylabel("Mean Action Magnitude")

    axs[2].set_title("Tracking Error (Angular Velocity)")
    axs[2].set_ylabel("Mean Tracking Error")

    axs[3].set_title("Episode Lengths")
    axs[3].set_ylabel("Steps per Episode")
    axs[3].set_xlabel("Episodes")

    for episode in range(0, episodes, EVAL_INTERVAL_EPISODES):
        print(f"type model is {type(model.policy)}")
        model.learn(total_timesteps=save_interval, reset_num_timesteps=False)

        # Evaluate model after every EVAL_INTERVAL_EPISODES episodes
        (
            avg_reward,
            avg_action_magnitude,
            avg_angular_velocity_error,
            avg_episode_length,
        ) = evaluate_model(env, model, num_episodes=10)

        # Store results
        model.logger.record("train/action_magnitude", avg_action_magnitude)
        model.logger.record("train/value_estimate", np.mean(avg_reward))
        model.logger.record("train/advantage", np.mean(avg_angular_velocity_error))
        model.logger.record("rollout/reward_std", np.std(avg_reward))

        model.logger.dump(step=episode + EVAL_INTERVAL_EPISODES)

        # Update live plots
        for i, data, ylabel in zip(
            range(4),
            [
                reward_history,
                action_magnitude_history,
                angular_velocity_error_history,
                episode_length_history,
            ],
            [
                "Average Reward",
                "Mean Action Magnitude",
                "Mean Tracking Error",
                "Steps per Episode",
            ],
        ):
            axs[i].clear()
            axs[i].plot(episode_numbers, data, marker="o", linestyle="-")
            axs[i].set_ylabel(ylabel)
            if i == 3:
                axs[i].set_xlabel("Episodes")

        plt.pause(0.1)

        model_filename = f"/opt/autorl_ws/models/{MODEL_BASENAME}_{START_TRAINING_FROM + episode + EVAL_INTERVAL_EPISODES}.zip"
        model.save(model_filename)
        print(f"Model saved: {model_filename}")

    model.save(model_save_path)
    print(f"Final model saved as: {model_save_path}")
    plt.ioff()
    plt.show()


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
    rclpy.init(args=args)
    env = gym.make("Autopilot-RL-v0")
    check_env(env)

    logger = rclpy.logging.get_logger("my_logger")
    logger.info("Starting PPO training")
    try:
        max_steps = env.spec.max_episode_steps
        num_episodes = TOTAL_EPISODES
        start_time = time.time()  # Store start time
        train_model(env, episodes=num_episodes, max_steps=max_steps)
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
