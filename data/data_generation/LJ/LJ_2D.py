import numpy as np
import matplotlib.pyplot as plt

# ---------- PARAMETERS ----------
np.random.seed(42)
n_particles = 5
dim = 2  
steps = 20000
dt = 0.005
mass = 1.0

# Lennard-Jones parameters
epsilon = 1.0
sigma = 1.0
cutoff = 2.5 * sigma  # common cutoff radius

box_size = 10.0

# ---------- INITIALIZE PARTICLES ----------
positions = np.random.rand(n_particles, dim) * box_size
velocities = (np.random.rand(n_particles, dim) - 0.5) * 0.1
accelerations = np.zeros_like(positions)

# Store history
trajectory = np.zeros((steps, n_particles, dim))


# ---------- FORCE CALCULATION (LJ) ----------
def compute_forces(pos):
    forces = np.zeros_like(pos)
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            rij = pos[j] - pos[i]
            # PBC
            rij = rij - box_size * np.round(rij / box_size)

            r = np.linalg.norm(rij)
            if r < cutoff and r > 1e-12:
                r6 = (sigma / r) ** 6
                r12 = r6 * r6
                f_mag = 24 * epsilon * (2 * r12 - r6) / r**2
                force = f_mag * rij

                forces[i] += force
                forces[j] -= force  # Newton's third law
    return forces


# ---------- VELOCITY VERLET INTEGRATION ----------
forces = compute_forces(positions)

for step in range(steps):

    trajectory[step] = positions
    positions += velocities * dt + 0.5 * forces / mass * dt**2
    positions %= box_size
    new_forces = compute_forces(positions)
    velocities += 0.5 * (forces + new_forces) / mass * dt
    forces = new_forces

# ---------- SAVE TRAJECTORY ----------
np.savez("verlet_lj_2d_5particles.npz", trajectory=trajectory, box_size=box_size)

print("Simulation complete. Trajectory saved to 'verlet_lj_2d_5particles.npz'.")
