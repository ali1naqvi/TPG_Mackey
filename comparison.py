import matplotlib.pyplot as plt
import numpy as np

# Function to read data from a text file
def read_data(file_path):
    with open(file_path, 'r') as file:
        data = [float(line.strip()) for line in file]
    return np.array(data)

# File paths (update these to the actual paths of your text files)
objective_file = 'mackey_glass_1100_samples.txt'
simulated_files = ['Simulation_2.txt', 'Simulation_3.txt']

objective_data = read_data(objective_file)[800:]
simulated_data = [read_data(file)[800:] for file in simulated_files]

# Plot data
plt.plot(objective_data, label='Objective', color='black')

colors = ['blue', 'green']
for i, data in enumerate(simulated_data):
    plt.plot(data, label=f'Simulated {i+1}', color=colors[i])

plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Comparison of Simulated Data vs Objective Data')
plt.legend()
plt.grid(True)
plt.show()
