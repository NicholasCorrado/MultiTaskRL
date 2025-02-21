import numpy as np
import matplotlib.pyplot as plt

# Load the npz file
data = np.load("evaluations.npz")  # Ensure the correct path

# Extract timestep and return values
timesteps = data["timestep"]
returns = data["return"]

# Plot
plt.figure(figsize=(8, 5))
plt.plot(timesteps, returns, marker='o', linestyle='-')
plt.xlabel("Timestep")
plt.ylabel("Return")
plt.title("Return vs. Timestep")
plt.grid(True)
plt.show()