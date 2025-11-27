import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from threading import Event
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy


class SimTimeHandler:
    def __init__(self, node, topic_name="/SimTime"):
        self.node = node
        self.topic_name = topic_name
        self.clock_data = None
        self.data_event = Event()

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.subscription = self.node.create_subscription(
            Clock, self.topic_name, self.clock_listener_callback, qos_profile
        )
        self.node.get_logger().info(f"Subscribed to topic: {self.topic_name}")

    def clock_listener_callback(self, msg):
        self.clock_data = msg
        self.data_event.set()  # Set the event to signal that data is available

    def wait_for_clock_data(self, timeout=1.0):
        if not self.data_event.wait(timeout):
            self.node.get_logger().error(
                "IMU data not received within the timeout period."
            )
            raise TimeoutError("Clock data not received within the timeout period.")
        self.data_event.clear()
        return self.clock_data
