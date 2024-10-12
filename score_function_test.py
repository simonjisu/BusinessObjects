import matplotlib.pyplot as plt
import numpy as np

plt.style.use('default')

N = 8
a_values = np.arange(0, N, 1)  # Example range from 0 to 9
b_values = np.arange(0, N, 1)  # Example range from 0 to 9

# Create a matrix to hold the results
result_matrix = np.zeros((len(a_values), len(b_values)))
def normalize_values(x, min_value=0, max_value=5):
    normalized = (x - min_value) / (max_value - min_value)
    return normalized

def tanh(x: np.ndarray, k: float):
    normalized = normalize_values(x, max_value=k)
    return np.tanh(np.log(1+normalized.sum()))

def archtan(x: np.ndarray, k: float):
    normalized = normalize_values(x, max_value=k)
    return (2 / np.pi) * np.arctan(normalized.sum())

def sigmoid(x: np.ndarray, k: float=0.1):
    # sigmoid function
    normalized = normalize_values(x, min_value=-k, max_value=k)
    return 1 / (1 + np.exp(-normalized.sum()))

def exponential(x: np.ndarray, k: float=0.1):
    normalized = normalize_values(x, max_value=k)
    return np.exp(-normalized.sum())

func_name = 'tanh'
func = {
    'tanh': tanh,
    'archtan': archtan,
    'sigmoid': sigmoid,
    'exponential': exponential
}[func_name]

result_matrices = {}

ks = [1, 2, 5, 6]
# Populate the matrix with the new function values
for k in ks:
    result_matrices[k] = np.zeros((len(a_values), len(b_values)))
    for i, a in enumerate(a_values):
        for j, b in enumerate(b_values):
            result_matrices[k][i, j] = func(np.array([a, b]), k)

# Plot the matrix for the new function
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.tight_layout(pad=5.0)

for i, (k, ax) in enumerate(zip(ks, axes.flatten())):
    # contourf
    # contour = ax.contourf(b_values, a_values, result_matrices[k], levels=50, cmap='viridis')
    matshow = ax.matshow(result_matrices[k], cmap='coolwarm')
    ax.set_title(f'k = {k}')
    ax.set_xlabel('b values')
    ax.set_ylabel('a values')
    ax.invert_yaxis()
    # add numbering 
    for (i, j), val in np.ndenumerate(result_matrices[k]):
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black')
    # Add color bar for each plot
    cbar = fig.colorbar(matshow, ax=ax, orientation='vertical', pad=0.05)
    cbar.set_label('Mapped Value')

plt.show()