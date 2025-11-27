from actuator_msgs.msg import Actuators
import rclpy
from rclpy.node import Node


class MotorHandler:
    def __init__(self, node, topic_name="/MOTOR_SPEED"):
        self.node = node  
        self.topic_name = topic_name
        self.publisher = self.node.create_publisher(Actuators, self.topic_name, 10)

    def publish_motor_speeds(self, motor_speeds):
        """Publishes motor speeds."""
        motor_speed_msg = Actuators()
        motor_speed_msg.velocity = motor_speeds
        # motor_speed_msg.header.stamp = self.node.get_clock().now().to_msg()
        # motor_speed_msg.header.frame_id = "base_link"
        self.publisher.publish(motor_speed_msg)
