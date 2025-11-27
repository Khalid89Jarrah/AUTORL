import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from threading import Event
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy


class IMUHandler:
    def __init__(self, node, topic_name="/IMU_SIGNAL"):
        self.node = node
        self.topic_name = topic_name
        self.imu_data = None
        self.data_event = Event()

        self.subscription = self.node.create_subscription(
            Imu,
            self.topic_name,
            self.IMU_listener_callback,
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            ),
        )
        self.node.get_logger().info(f"Subscribed to topic: {self.topic_name}")

    def IMU_listener_callback(self, msg):
        self.imu_data = msg
        self.data_event.set()

    def wait_for_imu_data(self, timeout=1.0):
        if not self.data_event.wait(timeout):
            self.node.get_logger().error(
                "\033[31mIMU data not received within the timeout period.\033[0m"
            )
            raise TimeoutError("IMU data not received within the timeout period.")
        self.data_event.clear()
        return self.imu_data
