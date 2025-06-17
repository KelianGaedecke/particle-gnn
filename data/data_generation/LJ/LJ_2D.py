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
forces_history = np.zeros((steps, n_particles, dim))
velocities_history = np.zeros((steps, n_particles, dim))
potential_energies = np.zeros(steps)

# ---------- FORCE + ENERGY CALCULATION ----------
def compute_forces_and_energy(pos):
    forces = np.zeros_like(pos)
    energy = 0.0
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            rij = pos[j] - pos[i]
            rij = rij - box_size * np.round(rij / box_size)
            r = np.linalg.norm(rij)

            if r < cutoff and r > 1e-12:
                r6 = (sigma / r) ** 6
                r12 = r6 * r6
                f_mag = 24 * epsilon * (2 * r12 - r6) / r**2
                force = f_mag * rij

                forces[i] += force
                forces[j] -= force

                energy += 4 * epsilon * (r12 - r6)
    return forces, energy

# ---------- VELOCITY VERLET INTEGRATION ----------
forces, pot_energy = compute_forces_and_energy(positions)

for step in range(steps):
    trajectory[step] = positions
    forces_history[step] = forces
    velocities_history[step] = velocities
    potential_energies[step] = pot_energy

    positions += velocities * dt + 0.5 * forces / mass * dt**2
    positions %= box_size
    new_forces, pot_energy = compute_forces_and_energy(positions)
    velocities += 0.5 * (forces + new_forces) / mass * dt
    forces = new_forces

# ---------- SAVE ALL DATA ----------
np.savez("verlet_lj_2d_5particles.npz", 
         trajectory=trajectory, 
         forces=forces_history, 
         velocities=velocities_history,
         potential_energy=potential_energies,
         box_size=box_size)

print("Simulation complete. Trajectory, forces, velocities, and potential energy saved.")

data = np.load("verlet_lj_2d_5particles.npz")
print(data.files)
