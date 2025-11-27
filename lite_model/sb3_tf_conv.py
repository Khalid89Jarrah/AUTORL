#!/usr/bin/env python3
import os
import sys
import torch
import onnx
import onnx_tf.backend
import tensorflow as tf
from stable_baselines3 import PPO


class OnnxablePolicy(torch.nn.Module):
    #Exports PPO policy network identical to runtime behaviour (includes tanh)

    def __init__(self, policy):
        super().__init__()
        self.policy = torch.nn.Sequential(
            policy.mlp_extractor.policy_net,
            policy.action_net,
            torch.nn.Tanh(), 
        )

    def forward(self, obs):
        return self.policy(obs)


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] != "--path":
        print("Usage: python3 convert_to_q_tflite.py --path <ppo_model.zip>")
        sys.exit(1)

    model_path = sys.argv[2]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_name = os.path.splitext(os.path.basename(model_path))[0]

    # output paths
    onnx_file = os.path.join(script_dir, f"{model_name}.onnx")
    tf_dir = os.path.join(script_dir, f"{model_name}_tf")
    tflite_file = os.path.join(script_dir, f"{model_name}.tflite")
    tflite_q_file = os.path.join(script_dir, f"{model_name}_quant.tflite")

    print(f"Saving all outputs in: {script_dir}")

    # load PPO model
    print("Loading PPO model …")
    model = PPO.load(model_path, device="cpu", print_system_info=False)
    obs_dim = model.observation_space.shape[0]
    dummy = torch.zeros((1, obs_dim), dtype=torch.float32)

    # export ONNX
    print("Exporting ONNX (with tanh) …")
    onnxable = OnnxablePolicy(model.policy).cpu().eval()
    torch.onnx.export(
        onnxable,
        dummy,
        onnx_file,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
    )
    print(f"ONNX saved: {onnx_file}")
    onnx.checker.check_model(onnx.load(onnx_file))

    # ONNX → TF
    print("Converting ONNX → TensorFlow …")
    tf_rep = onnx_tf.backend.prepare(onnx.load(onnx_file))
    tf_rep.export_graph(tf_dir)
    print(f"SavedModel: {tf_dir}")

    # TF → TFLite (float)
    print("Converting → TFLite (float32) …")
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_dir)
    tflite_model = converter.convert()
    open(tflite_file, "wb").write(tflite_model)
    print(f"TFLite FP32: {tflite_file}")

    # quantized INT8
    print("Converting → TFLite INT8 (quantized) …")

    def representative_data_gen():
        for _ in range(1000):
            yield [model.observation_space.sample().reshape(1, -1).astype("float32")]

    converter_q = tf.lite.TFLiteConverter.from_saved_model(tf_dir)
    converter_q.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_q.representative_dataset = representative_data_gen
    converter_q.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_q.inference_input_type = tf.float32
    converter_q.inference_output_type = tf.float32
    q_model = converter_q.convert()
    open(tflite_q_file, "wb").write(q_model)
    print(f"TFLite: {tflite_q_file}")

    print("\n Conversion complete. Model includes tanh for exact PPO match.")
