# AUTORL Docker Environment

A complete Docker environment for running AUTORL using ROS 2 Jazzy, Gazebo Harmonic, and reinforcement learning tools.

---

## Included Components

This Docker image installs and configures:

- **ROS 2 (Jazzy)**
- **Gazebo Harmonic**
- **ros_gz bridge**
- **Stable-Baselines3 and TensorFlow**
- **Python virtual environment**
- **All system and simulation dependencies**

---

## Build the Docker Image

To build the Docker image, run:

```bash
    docker build --network=host -f docker/Dockerfile --progress=plain -t autorl:latest .
```

## Run the docker images:
```bash
   . docker/run_docker.sh IMAGE_ID --src_path AUTORL_src_path --model_path AUTORL_model_path
```

 