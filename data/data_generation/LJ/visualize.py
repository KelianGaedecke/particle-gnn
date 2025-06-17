import numpy as np
import matplotlib.pyplot as plt

# Load the trajectory from the .npz file
data = np.load("verlet_lj_2d_5particles.npz")
trajectory = data["trajectory"]

# Extract metadata
steps, n_particles, dims = trajectory.shape
box_size = 10.0 

# Animate the trajectory
for t in range(0, steps, 50):
    plt.clf()
    plt.xlim(0, box_size)
    plt.ylim(0, box_size)
    plt.title(f"Step {t}")
    plt.scatter(*trajectory[t].T, c='blue')
    plt.pause(0.05)

plt.show()
