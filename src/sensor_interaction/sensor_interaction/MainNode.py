import rclpy
import threading
from rclpy.node import Node
from sensor_interaction.imu_handler import IMUHandler
from sensor_interaction.sim_time_handler import (
    SimTimeHandler,
)
from sensor_interaction.simulation_reset_handler import (
    SimulationResetHandler,
)
from sensor_interaction.simulation_step_handler import (
    SimulationStepHandler,
)
from sensor_interaction.motor_handler import MotorHandler
from sensor_interaction.psude_inverse import PseudoInverse
from sensor_interaction.sequentialdesaturationmotoroutput import (
    SequentialDesaturationMotorOutput,
)
import xml.etree.ElementTree as ET
import numpy as np
import yaml
from pathlib import Path


class MainNode(Node):
    def __init__(self):
        super().__init__("main_node")
        self.executor_thread = threading.Thread(target=self.run_executor, daemon=True)
        self.executor_running = True
        self.executor_thread.start()
        self.imu_handler = IMUHandler(self)
        self.sim_time_handler = SimTimeHandler(self)
        self.sim_reset_handler = SimulationResetHandler(self)
        self.sim_step_handler = SimulationStepHandler(self)
        self.motor_handler = MotorHandler(self)
        self.psude_inverse = PseudoInverse()
        self.sequential_desaturation = SequentialDesaturationMotorOutput()
        self.params = self.load_yaml()

    def load_yaml(self):
        current_dir = Path(__file__).resolve().parent
        yaml_path = current_dir.parent / "resources" / "autorl_drone" / "gz_x500.yaml"

        with open(yaml_path, "r") as f:
            return yaml.safe_load(f)

    def run_executor(self):
        while self.executor_running and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)

    def stop_executor(self):
        self.executor_running = False
        if self.executor_thread.is_alive():
            self.executor_thread.join(timeout=1.0)

    def destroy_node(self):
        self.stop_executor()
        return super().destroy_node()

    def read_imu(self):
        imu_data = self.imu_handler.wait_for_imu_data(10.2)
        return imu_data

    def get_sim_time(self):
        # self.get_logger().info("Attempting to get simulation time.")
        try:
            clock_data = self.sim_time_handler.wait_for_clock_data(
                10.2
            )  # Wait for clock data
            sim_time_sec = clock_data.clock.sec
            sim_time_nanosec = clock_data.clock.nanosec
            total_sim_time = sim_time_sec + sim_time_nanosec * 1e-9
            # self.get_logger().info(f"Simulation Time: {total_sim_time:.9f} seconds")
            return total_sim_time
        except TimeoutError as e:
            self.get_logger().error(f"Failed to get simulation time: {e}")
            return None

    def reset_simulation(self):
        """Reset the simulation using SimulationResetHandler."""
        # self.get_logger().info("Attempting to reset the simulation...")
        try:
            success = self.sim_reset_handler.reset_simulation(timeout=60)
        except TimeoutError as e:
            self.get_logger().error(f"Simulation reset timed out: {e}")

    def perform_simulation_step(self, steps=1):
        """Perform a simulation step using the SimulationStepHandler."""
        # self.get_logger().info(f"Attempting to perform {steps} simulation step(s)...")
        try:
            success = self.sim_step_handler.perform_simulation_step(
                steps=steps, timeout=60.0
            )
        except TimeoutError as e:
            self.get_logger().error(f"Simulation step timed out: {e}")

    def publish_motor_commands(self, motor_commands):
        try:
            self.motor_handler.publish_motor_speeds(motor_commands)
            # self.get_logger().info(f"Published motor commands: {motor_commands}")
        except Exception as e:
            self.get_logger().error(f"Failed to publish motor commands: {e}")

    def read_param_element(self, param1):
        return self.params.get(param1, None)

    def read_motor_specs(self):  # Updated function to read from YAML
        rotors = []
        for rotor_key, rotor_values in self.params["rotors"].items():
            rotor = {
                "motor_constant": rotor_values["KM"],
                "position": np.array([rotor_values["PX"], rotor_values["PY"], 0]),
                "axis": np.array(
                    [rotor_values["AX"], rotor_values["AY"], rotor_values["AZ"]]
                ),
                "ct": rotor_values["CT"],
            }
            rotors.append(rotor)
        return {"rotors": rotors}  # Return rotor specs

    def compute_effectiveness_matrix(self, specs):  # Use YAML-based specs
        rotors = specs["rotors"]
        num_rotors = len(rotors)
        effectiveness_matrix = np.zeros((6, num_rotors))

        for i, rotor in enumerate(rotors):
            axis = rotor["axis"].astype(np.float64)
            axis_norm = np.linalg.norm(axis)
            if axis_norm > np.finfo(float).eps:
                axis /= axis_norm
            position = rotor["position"].astype(np.float64)
            ct = rotor["ct"]
            km = rotor["motor_constant"]
            thrust = ct * axis
            moment = ct * np.cross(position, axis) - ct * km * axis
            effectiveness_matrix[:3, i] = moment
            effectiveness_matrix[3:, i] = thrust
        return effectiveness_matrix

    def compute_psudo_inverse(self, effectiveness_matrix):
        return self.psude_inverse.update_pseudo_inverse_full(effectiveness_matrix)

    def compute_sequential_desaturation(self, rate_control, psudo_inverse_matrix):
        return self.sequential_desaturation.mixAirmodeDisabled(
            rate_control, psudo_inverse_matrix
        )
