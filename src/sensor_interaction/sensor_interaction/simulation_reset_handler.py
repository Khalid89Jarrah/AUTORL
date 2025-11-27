from rclpy.node import Node
from ros_gz_interfaces.srv import (
    ControlWorld,
)
from ros_gz_interfaces.msg import (
    WorldReset,
    WorldControl,
)
from threading import Event


class SimulationResetHandler:
    def __init__(self, node):
        self.node = node
        self.client = self.node.create_client(ControlWorld, "/world/default/control")
        self.response_event = Event()  # Event to block until response is received
        self.response = None

    def reset_simulation(self, timeout=10.0):
        # Sends a reset command and blocks until the response is received.
        self.response_event.clear()
        self.response = None

        request = ControlWorld.Request()
        world_reset = WorldReset()
        world_reset.all = True  # Reset the entire world
        world_control = WorldControl()
        world_control.reset = world_reset
        world_control.pause = True
        world_control.seed = 0
        request.world_control = world_control

        # self.node.get_logger().info("Sending simulation reset command...")

        # Perform the asynchronous call
        future = self.client.call_async(request)
        future.add_done_callback(self._handle_response)

        # Block until response is received or timeout occurs
        if not self.response_event.wait(timeout):
            raise TimeoutError(
                "Simulation reset response not received within the timeout period."
            )

        if self.response and self.response.success:
            # self.node.get_logger().info("Simulation reset completed successfully.")
            return True
        else:
            self.node.get_logger().warn("Simulation reset failed.")
            return False

    def _handle_response(self, future):
        """Handles the response after resetting the simulation."""
        try:
            self.response = future.result()  # Get the response
        except Exception as e:
            self.node.get_logger().error(f"Error in resetting simulation: {e}")

        # Set the event to unblock waiting thread
        self.response_event.set()
