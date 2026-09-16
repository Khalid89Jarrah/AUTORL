# AUTORL

Hybrid Reinforcement Learning (RL) training of the attitude controller using ROS 2 and Gazebo.

> **Note:** This repository is for simulation and research purposes only. It is not
> validated for real flight or integration into a flight control stack.

---

## Project Structure

This project integrates:
- **ROS 2 (Jazzy)**
- **Gazebo Harmonic**
- **Stable-Baselines3**

---

## Build the Workspace

To build and source the ROS 2 workspace, run:

```bash
cd /opt/autorl_ws/src
. colcon_build

```
## Launch training
Run this command to run the training 
```
ros2 launch sensor_interaction node_launch.py algorithm:=ppo gui:=false mode:=training
```
To avoid buffering the std output run
```
PYTHONUNBUFFERED=1 ros2 launch sensor_interaction node_launch.py gui:=true mode:=training
```

## Lanuch log
Run this command to check the tensorboard log
```
tensorboard --logdir=ppo_logs --port=6006
```
 And run this in ur browser http://localhost:6006/

## Third-Party Licenses

This project includes a modified version of a PX4 model licensed under the
BSD-3 Clause License. It also contains my own Python reimplementations based on
my understanding of the algorithms used in the PX4 multicopter mixer and
control-allocation system.

See licenses/LICENSE.PX4 for full license details. The x500 model assets are licensed under BSD-3 by Rudis Laboratories; see the
LICENSE files in src/sensor_interaction/resources/autorl_drone/.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{jarrah2026autorl,
  title={AutoRL: A Tightly Synchronized ROS2--Gazebo Pipeline for Offline-Trained Reinforcement Learning-Based Multirotor Attitude Control},
  author={Jarrah, Khaled and Rawashdeh, Osamah},
  journal={Aerospace},
  volume={13},
  number={9},
  pages={825},
  year={2026},
  publisher={MDPI}
}
```