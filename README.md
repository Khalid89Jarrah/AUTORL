# AUTORL

Hybrid Reinforcement Learning (RL) training of the attitude controller using ROS 2 and Gazebo.

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

See licenses/LICENSE.PX4 for full license details.