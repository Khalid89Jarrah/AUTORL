import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from threading import Event, Lock
from collections import deque
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy


class IMUHandler:
    def __init__(self, node, topic_name="/IMU_SIGNAL"):
        self.node = node
        self.topic_name = topic_name
        self.imu_data = None          # latched: last message seen (async readers)
        self.data_event = Event()
        # The callback used to overwrite a single slot. If the executor thread
        # delivered message N+1 between wait() returning and the reader taking
        # the value, message N was silently lost. Measured at ~1 per 10,000
        # steps, and unaffected by DDS queue depth. A queue removes it.
        self._queue = deque(maxlen=200)
        self._lock = Lock()

        self.subscription = self.node.create_subscription(
            Imu,
            self.topic_name,
            self.IMU_listener_callback,
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=50,
            ),
        )
        self.node.get_logger().info(f"Subscribed to topic: {self.topic_name}")

    def IMU_listener_callback(self, msg):
        with self._lock:
            self._queue.append(msg)
            self.imu_data = msg
        self.data_event.set()

    def wait_for_imu_data(self, timeout=30.0):
        if not self.data_event.wait(timeout):
            self.node.get_logger().error(
                "\033[31mIMU data not received within the timeout period.\033[0m"
            )
            raise TimeoutError("IMU data not received within the timeout period.")
        with self._lock:
            msg = self._queue.popleft() if self._queue else self.imu_data
            if not self._queue:
                self.data_event.clear()
        return msg

    def flush(self):
        """Discard every buffered message. Used at episode reset."""
        with self._lock:
            self._queue.clear()
            self.imu_data = None
            self.data_event.clear()
