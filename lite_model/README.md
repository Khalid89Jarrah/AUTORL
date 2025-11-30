# SB3 to TFLite Conversion and Quantization

This README contains all the steps required to set up the environment, convert a Stable Baselines3 (SB3) model to TensorFlow Lite (TFLite), and test the quantized model.

---

## Commands

> **Note:**  
> As you finished the model and saved it in your workspace directory, create a new
> virtual environment on the host to isolate it.  
> Alternatively, you can mount or copy the `lite_model` into the Docker container
> and run it there.

> **Note:** 
>If IPv6 is advertised by the network but IPv6 TCP traffic is not routed, causing connections to hang, Run this:
> ```bash
> echo "precedence ::ffff:0:0/96 100" >> /etc/gai.conf
> ```


```bash
# Update system and install Python 3.9
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.9 python3.9-venv python3.9-dev

# Create and activate virtual environment
cd /path/to/your/project
python3.9 -m venv sb3_venv_v_3.9
source sb3_venv_v_3.9/bin/activate
pip install --upgrade pip

# Install core dependencies
pip install gymnasium==1.1.1 pyglet==2.1.11 torch==2.8.0 torchsummary==1.5.1 stable-baselines3==2.7.0 \
  tensorflow==2.10.0 onnx==1.12.0 onnx-tf==1.9.0 protobuf==3.19.6 \
  --extra-index-url https://google-coral.github.io/py-repo/ pycoral==2.0.0 \
  --extra-index-url https://google-coral.github.io/py-repo/ tflite-runtime==2.5.0.post1

# Compatibility fixes
pip install "numpy==1.26.4"
pip install "shimmy==2.0.0"
pip install "gymnasium[box2d]"
# box2d-py installed as: box2d-py==2.3.5

# Convert SB3 model to TFLite
python3 /path/to/lite_model/sb3_tf_conv.py --path models/ppo_model.zip

# Test quantized TFLite model
# This script compares the original SB3 PPO policy outputs against the quantized TFLite 
# model across thousands of random observations. It reports accuracy metrics and 
# inference latency to #verify that the TFLite model is equivalent and ready for deployment.

python3 /path_to/lite_model/sb3_tf_conv_test.py --zip models/ppo_model.zip --tflite /path_to/lite_model/ppo_model_episode_quant.tflite --n 100000
```



## SB3 Model Inspection Tools

These scripts extract and inspect the architecture, weights, and training metadata of a Stable-Baselines3 PPO model.
```bash
# Run architecture/weights extractor
python3 lite_model/print_architecture.py --model your_model.zip
```

To print all training related information (hyperparameters, normalization stats, spaces, etc.) into a readable file do
```bash
# Run training info extractor
python3 lite_model/print_training_info.py --model your_model.zip
```