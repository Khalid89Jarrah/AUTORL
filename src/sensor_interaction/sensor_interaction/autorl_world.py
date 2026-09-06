import numpy as np
import gymnasium as gym
from gymnasium import spaces


class Auto_RL(gym.Env):
    def __init__(self, main_node, simulation_step_time=0.004):
        self.main_node = main_node
        self.simulation_step_time = simulation_step_time

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        self.previous_control_effort = np.zeros(3, dtype=np.float32)
        self.previous_angular_velocity = np.zeros(3, dtype=np.float32)
        self.previous_angular_velocity_2 = np.zeros(3, dtype=np.float32)
        self.stable_duration = 0.0

        specs = self.main_node.read_motor_specs()
        self.effectiveness_matrix = self.main_node.compute_effectiveness_matrix(specs)
        self.inverse_effectiveness_matrix = self.main_node.compute_psudo_inverse(
            self.effectiveness_matrix
        )

        # self.Hover_thrust = self.main_node.read_param_element('MPC_THR_HOVER')
        self.Hover_thrust = -0.9

    def reset(self, seed=None, options=None):
        eval_setpoint = options.get("eval_setpoint") if options else None
        super().reset(seed=seed, options=options)

        self.main_node.reset_simulation()

        self.stable_duration = 0.0  # Reset stable duration tracker

        # Initialize observation components
        angular_velocity_error = np.zeros(3, dtype=np.float32)
        actual_angular_velocity = np.zeros(3, dtype=np.float32)
        imu_linear_accelerations = np.zeros(3, dtype=np.float32)
        self.previous_control_effort = np.zeros(
            3, dtype=np.float32
        )  # For torques [\u03c4x, \u03c4y, \u03c4z]
        self.previous_angular_velocity = np.zeros(
            3, dtype=np.float32
        )  # Initialize previous angular velocity
        self.previous_angular_velocity_2 = np.zeros(
            3, dtype=np.float32
        )  # Initialize previous angular velocity

        if eval_setpoint is not None:
            self.angular_velocity_sp = eval_setpoint
        else:
            self.angular_velocity_sp = self.np_random.normal(
                loc=0.0, scale=0.3, size=3
            ).astype(np.float32)

        obs = np.concatenate(
            [
                angular_velocity_error,
                actual_angular_velocity,
                imu_linear_accelerations,
                self.previous_control_effort,
                self.previous_angular_velocity_2,
            ]
        ).astype(np.float32)

        info = {
            "action_space": self.action_space.shape,
            "observation_space": self.observation_space.shape,
            "additional_info": "Environment reset completed successfully",
        }
        return obs, info

    def step(self, action):
        try:
            # TODO:check thrustX and thrustY
            self.controlallocate = np.concatenate(
                (np.array(action), np.array([0.000000, 0.000000, self.Hover_thrust]))
            )
            rotor_speeds = self.main_node.compute_sequential_desaturation(
                self.controlallocate, self.inverse_effectiveness_matrix
            )
            # Publish motor commands
            self.main_node.publish_motor_commands(rotor_speeds)

            # Perform simulation step
            self.main_node.perform_simulation_step(1)

            # Read IMU data
            imu_msg = self.main_node.read_imu()
            angular_velocity = np.array(
                [
                    imu_msg.angular_velocity.x,
                    imu_msg.angular_velocity.y,
                    imu_msg.angular_velocity.z,
                ],
                dtype=np.float32,
            )

            imu_acceleration = np.array(
                [
                    imu_msg.linear_acceleration.x,
                    imu_msg.linear_acceleration.y,
                    imu_msg.linear_acceleration.z,
                ],
                dtype=np.float32,
            )

            # Compute angular velocity error
            angular_velocity_error = self.angular_velocity_sp - angular_velocity

            # Track stability duration
            stability_threshold = 0.05  # rad/s
            if np.linalg.norm(angular_velocity_error) <= stability_threshold:
                self.stable_duration += self.simulation_step_time
            else:
                self.stable_duration = 0.0

            # Compute reward
            reward = self._compute_reward(
                angular_velocity_error,
                angular_velocity,
                imu_acceleration,
                self.previous_control_effort,
                action,
                self.previous_angular_velocity,
                self.previous_angular_velocity_2,
                self.angular_velocity_sp,
            )

            self.previous_control_effort = action

            self.previous_angular_velocity_2 = self.previous_angular_velocity

            self.previous_angular_velocity = angular_velocity

            # Check termination
            terminated = self._check_termination(
                angular_velocity,
                angular_velocity_error,
                self.stable_duration,
                max_stable_duration=2.0,
            )

            # Check truncation
            sim_time = self.main_node.get_sim_time()
            if sim_time is None:
                raise TimeoutError("Simulation clock unavailable")
            truncated = sim_time > 3.0

            # Construct observation
            obs = np.concatenate(
                [
                    angular_velocity_error,
                    angular_velocity,
                    imu_acceleration,
                    self.previous_control_effort,
                    self.previous_angular_velocity_2,
                ]
            ).astype(np.float32)

            return obs, reward, terminated, truncated, {}

        except TimeoutError:
            self.main_node.get_logger().warning(
                "Timeout occurred during step execution. Skipping this action."
            )
            obs, info = self.reset()
            return obs, 0.0, False, False, info

    def _compute_reward(
        self,
        angular_velocity_error,
        actual_angular_velocity,
        imu_acceleration,
        previous_control_effort,
        current_control_effort,
        previous_angular_velocity,
        previous_angular_velocity_2,
        angular_velocity_setpoint,
    ):
        # --- Tracking error---
        R_error = -0.5 * np.linalg.norm(angular_velocity_error) ** 2
        R_shaped = -0.1 * np.linalg.norm(angular_velocity_error)

        # --- Oscillation penalty ---
        angular_accel = (
            actual_angular_velocity
            - 2 * previous_angular_velocity
            + previous_angular_velocity_2
        )
        R_oscillation = -1.0 * np.linalg.norm(angular_accel) ** 2

        # --- Control effort smoothness ---
        delta_u = current_control_effort[:3] - previous_control_effort
        R_effort_smooth = -0.75 * np.linalg.norm(delta_u) ** 2  # was -0.25

        # --- Energy usage ---
        R_effort_energy = (
            -0.40 * np.linalg.norm(current_control_effort[:3]) ** 2
        )  # was -0.063

        # --- Disturbance robustness ---
        R_disturbance = -0.02 * np.linalg.norm(imu_acceleration)

        # --- Overshoot penalty ---
        sgn = np.sign(angular_velocity_setpoint)
        signed_error = sgn * (actual_angular_velocity - angular_velocity_setpoint)
        overshoot = np.clip(signed_error, 0.0, None)
        R_overshoot = -2.0 * np.linalg.norm(overshoot) ** 2

        # --- Tracking band reward ---
        error_norm = np.linalg.norm(angular_velocity_error)
        epsilon = 0.06
        R_band = +0.5 * np.exp(-((error_norm / epsilon) ** 2))  # was +1.0

        return (
            R_error
            + R_shaped
            + R_oscillation
            + R_effort_smooth
            + R_effort_energy
            + R_disturbance
            + R_overshoot
            + R_band
        )

    def _check_termination(
        self,
        angular_velocity,
        angular_velocity_error,
        stable_duration,
        max_stable_duration,
    ):
        # TODO: check the angular velocity limits and the stability threshold
        # Terminate if angular velocity exceeds safety limits
        angular_velocity_limit = 10.0  # rad/s
        if np.linalg.norm(angular_velocity) > angular_velocity_limit:
            return True

        # Terminate if the rotor remains in a steady state for long
        stability_threshold = 0.05  # rad/s
        if (
            np.linalg.norm(angular_velocity_error) <= stability_threshold
            and stable_duration >= max_stable_duration
        ):
            return True

        return False
