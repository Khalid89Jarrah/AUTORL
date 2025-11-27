# SB3 to TFLite Conversion and Quantization

This README contains all the steps required to set up the environment, convert a Stable Baselines3 (SB3) model to TensorFlow Lite (TFLite), and test the quantized model.

---

## Commands

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
pip install gymnasium pyglet torch torchsummary stable-baselines3 stable-baselines3[extra] \
  tensorflow==2.10.0 onnx==1.12.0 onnx-tf==1.9.0 protobuf==3.19.6 \
  --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=2.0 \
  --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime

# Compatibility fixes
pip install "numpy<2"
pip install "shimmy>=2.0"
pip install "gymnasium[box2d]"

# Convert SB3 model to TFLite
python3 /path/to/lite_model/sb3_tf_conv.py --path models/ppo_model.zip

# Test quantized TFLite model
python3 /path_to/lite_model/sb3_tf_conv_test.py --zip models/ppo_model.zip --tflite ppo_model_episode_quant.tflite --n 100000
```



## SB3 Model Inspection Tools

These scripts extract and inspect the architecture, weights, and training metadata of a Stable-Baselines3 PPO model.
```bash
# Run architecture/weights extractor
python3 architecture.py --model your_model.zip
```

To print all training related information (hyperparameters, normalization stats, spaces, etc.) into a readable file do
```bash
# Run training info extractor
python3 training_info.py --model your_model.zip
```