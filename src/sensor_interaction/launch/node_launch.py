import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    AppendEnvironmentVariable,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch.actions import LogInfo


def generate_launch_description():
    pkg_ros_gz_sim = get_package_share_directory("ros_gz_sim")
    resource_path = (
        get_package_share_directory("sensor_interaction") + "/resources/autorl_drone/"
    )
    print(f"resources : {resource_path}")

    set_gazebo_resource_path_env = AppendEnvironmentVariable(
        "GZ_SIM_RESOURCE_PATH", resource_path
    )

    log_resource_path = LogInfo(msg=f"GZ_SIM_RESOURCE_PATH is set to: {resource_path}")

    declare_gui_argument = DeclareLaunchArgument(
        "gui", default_value="false", description="Launch with GUI (true/false)"
    )
    declare_algo_argument = DeclareLaunchArgument(
        "algorithm",
        default_value="ppo",
        description="Choose the RL algorithm: ppo or pid",
    )
    declare_mode_argument = DeclareLaunchArgument(
        "mode", default_value="training", description="Mode: training or evaluate"
    )
    declare_seed_argument = DeclareLaunchArgument(
        "seed",
        default_value="0",
        description="Random seed for training (Campaign A: 0..4)",
    )
    declare_timesteps_argument = DeclareLaunchArgument(
        "timesteps",
        default_value="0",
        description="Total training timesteps. Must be > 0; training aborts otherwise.",
    )

    # Choose GUI or No-GUI mode
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={"gz_args": "autorl_drone.sdf"}.items(),
        condition=IfCondition(
            PythonExpression(["'", LaunchConfiguration("gui"), "' == 'true'"])
        ),
    )

    gz_sim_no_gui = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, "launch", "gz_server.launch.py")
        ),
        launch_arguments={"world_sdf_file": "autorl_drone.sdf"}.items(),
        condition=IfCondition(
            PythonExpression(["'", LaunchConfiguration("gui"), "' == 'false'"])
        ),
    )

    enable_colorized_logs = SetEnvironmentVariable("RCUTILS_COLORIZED_OUTPUT", "1")

    # Define Training and Evaluation Nodes
    ppo_training_node = Node(
        package="sensor_interaction",
        executable="ppo_train.py",
        output="screen",
        arguments=[
            "--seed",
            LaunchConfiguration("seed"),
            "--timesteps",
            LaunchConfiguration("timesteps"),
        ],
        condition=IfCondition(
            PythonExpression(
                [
                    "'",
                    LaunchConfiguration("algorithm"),
                    "' == 'ppo' and '",
                    LaunchConfiguration("mode"),
                    "' == 'training'",
                ]
            )
        ),
    )

    declare_model_path_argument = DeclareLaunchArgument(
        "model_path",
        default_value="/opt/autorl_ws/models/ppo_model.zip",
        description="Path to PPO model",
    )

    ppo_evaluation_node = Node(
        package="sensor_interaction",
        executable="evaluate_ppo.py",
        output="screen",
        parameters=[{"model_path": LaunchConfiguration("model_path")}],
        condition=IfCondition(
            PythonExpression(
                [
                    "'",
                    LaunchConfiguration("algorithm"),
                    "' == 'ppo' and '",
                    LaunchConfiguration("mode"),
                    "' == 'evaluate'",
                ]
            )
        ),
    )

    ppo_evaluation_determinstic = Node(
        package="sensor_interaction",
        executable="determinism_test.py",
        output="screen",
        condition=IfCondition(
            PythonExpression(
                [
                    "'",
                    LaunchConfiguration("algorithm"),
                    "' == 'ppo' and '",
                    LaunchConfiguration("mode"),
                    "' == 'test_determinism'",
                ]
            )
        ),
    )

    pid_evaluation_node = Node(
        package="sensor_interaction",
        executable="evaluate_pid.py",
        output="screen",
        condition=IfCondition(
            PythonExpression(
                [
                    "'",
                    LaunchConfiguration("algorithm"),
                    "' == 'pid' and '",
                    LaunchConfiguration("mode"),
                    "' == 'evaluate'",
                ]
            )
        ),
    )

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/world/default/model/x500/link/base_link/sensor/imu_sensor/imu@sensor_msgs/msg/Imu@gz.msgs.IMU",
            "/world/default/control@ros_gz_interfaces/srv/ControlWorld",
            "/x500/command/motor_speed@actuator_msgs/msg/Actuators@gz.msgs.Actuators",
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
        ],
        remappings=[
            (
                "/world/default/model/x500/link/base_link/sensor/imu_sensor/imu",
                "/IMU_SIGNAL",
            ),
            ("/x500/command/motor_speed", "/MOTOR_SPEED"),
            ("/clock", "/SimTime"),
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            declare_model_path_argument,
            enable_colorized_logs,
            set_gazebo_resource_path_env,
            ppo_evaluation_determinstic,
            log_resource_path,
            declare_gui_argument,
            declare_algo_argument,
            declare_mode_argument,
            declare_seed_argument,
            declare_timesteps_argument,
            gz_sim,
            gz_sim_no_gui,
            bridge,
            ppo_training_node,
            ppo_evaluation_node,
            pid_evaluation_node,
        ]
    )


"""
ros2 launch sensor_interaction node_launch.py algorithm:=ppo gui:=false mode:=test_determinism
ros2 launch sensor_interaction node_launch.py algorithm:=pid gui:=false mode:=evaluate
ros2 launch sensor_interaction node_launch.py algorithm:=ppo gui:=false mode:=evaluate model_path:="/opt/autorl_ws/models/ppo_model.zip"
ros2 launch sensor_interaction node_launch.py algorithm:=ppo gui:=false mode:=training seed:=0 timesteps:=2000000
"""
