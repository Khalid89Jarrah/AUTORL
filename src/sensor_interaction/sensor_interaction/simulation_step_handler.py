import rclpy
from rclpy.node import Node
from threading import Event
from ros_gz_interfaces.srv import ControlWorld
from ros_gz_interfaces.msg import WorldReset, WorldControl


class SimulationStepHandler:
    def __init__(self, node):
        self.node = node
        self.client = self.node.create_client(ControlWorld, "/world/default/control")
        self.response_event = Event()
        self.response = None

    def perform_simulation_step(self, steps=1, timeout=5.0):
        """Perform a simulation step and wait for the response."""
        # Prepare the request
        request = ControlWorld.Request()
        world_reset = WorldReset()
        world_reset.all = False
        world_reset.time_only = False
        world_reset.model_only = False

        world_control = WorldControl()
        world_control.reset = world_reset
        world_control.pause = True  # Pause after the step
        world_control.multi_step = steps  # Specify the number of steps
        request.world_control = world_control

        # self.node.get_logger().info(f"Sending request to perform {steps} simulation step(s).")

        future = self.client.call_async(request)
        future.add_done_callback(self._simulation_step_callback)

        # Wait until the response is received or timeout occurs
        if not self.response_event.wait(timeout):
            raise TimeoutError(
                "Simulation step response not received within the timeout period."
            )

        if self.response and self.response.success:
            # self.node.get_logger().info(f"Simulation step completed successfully.")
            return True
        else:
            self.node.get_logger().warn("Simulation step failed.")
            return False

    def _simulation_step_callback(self, future):
        """Callback function to handle the response after performing the simulation step."""
        try:
            self.response = future.result()  # Get the response
        except Exception as e:
            self.node.get_logger().error(f"Error during simulation step: {e}")

        # Set the event to unblock the waiting thread
        self.response_event.set()
