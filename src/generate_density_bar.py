import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 1. Configuration
    num_points = 10000
    width = 1280
    height = 180
    # Set seed for reproducibility
    np.random.seed(42)

    # 2. Generate Gradient Density
    # We want x-coordinates to be more frequent on the right (higher value)
    # Using a power function to create a non-linear density gradient
    x = np.random.uniform(0, 1, num_points)**0.5 * width
    y = np.random.uniform(0, 1, num_points) * height

    # 3. Plotting
    plt.figure(figsize=(8, 8), dpi=100)
    plt.scatter(x, y, s=0.5, c='black', edgecolors='none')  # , alpha=0.6

    # 4. Styling
    plt.title('Graded Density Point Pattern', fontsize=15)
    plt.xlim(0, width)
    plt.ylim(0, height)
    plt.axis('off') # Turn off axes for cleaner look
    plt.gca().set_aspect('equal', adjustable='box')

    plt.show()