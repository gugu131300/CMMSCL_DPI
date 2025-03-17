import numpy as np
import matplotlib.pyplot as plt

# Generate random data
total_fea_test = np.random.rand(100, 2)
total_labels = np.random.randint(0, 2, 100)
print(total_labels)

# Scatter plot
plt.figure(figsize=(8, 6))

# Plot points with colors representing labels
# colors = [(117/255, 179/255, 113/255) if label == 0 else (230/255, 50/255, 50/255) for label in total_labels]
colors = [(117/255, 179/255, 113/255) if label == 0 else (230/255, 50/255, 50/255) if label == 1 else (0, 0, 0) for label in total_labels]
plt.scatter(total_fea_test[:, 0], total_fea_test[:, 1], color=colors, s=9)

# Add legend to the plot
plt.legend(['Negative Samples', 'Positive Samples'], loc='upper right')

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Random Data Visualization')

plt.show()
