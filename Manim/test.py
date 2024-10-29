from manimlib import *
import numpy as np

class SimplePendulum(Scene):
    def construct(self):
        # Pendulum parameters
        pivot_point = UP * 2  # The fixed point where the pendulum is attached
        string_length = 3     # Length of the pendulum's string
        amplitude = PI / 6    # Maximum angular displacement (in radians)

        # Create the pendulum mass (bob)
        bob = Dot(color=RED).move_to(
            pivot_point + string_length * np.array([np.sin(amplitude), -np.cos(amplitude), 0])
        )

        # Create the string (line connecting pivot to bob)
        string = Line(pivot_point, bob.get_center(), color=BLUE)

        # Label
        label = Text("Simple Pendulum Oscillator", font_size=24).to_edge(UP)

        # Add objects to the scene
        self.add(string, bob, label)

        # Define the oscillation movement of the pendulum
        def pendulum_motion(angle):
            x = string_length * np.sin(angle)
            y = -string_length * np.cos(angle)
            return pivot_point + np.array([x, y, 0])

        # Create the animation
        def update_pendulum(obj, dt):
            # Use a sinusoidal oscillation to approximate the pendulum's path
            time = self.time * 2  # Adjust speed
            angle = amplitude * np.cos(time)
            new_position = pendulum_motion(angle)
            bob.move_to(new_position)
            string.put_start_and_end_on(pivot_point, bob.get_center())

        # Apply the updater to animate the motion
        self.add(bob, string)  # Ensures both objects stay in the scene
        self.add(label)
        bob.add_updater(update_pendulum)
        string.add_updater(lambda m: m.put_start_and_end_on(pivot_point, bob.get_center()))

        # Run the scene
        self.wait(8)  # Duration of the pendulum motion animation
