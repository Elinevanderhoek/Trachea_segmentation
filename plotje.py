
## Heatmap of correlations for train and test data
## Eline van der Hoek
## 14-11-2024

## Input: Correlation scores
## Output: heatmap (image) displaying 'good' scores as green and 'bad' scores as red

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Train data
heatmap_data_R = {
    "Metric": ["Ratio", "Roundness", "dA", "Area", "dB", "Perimeter"],
    "U-Net": [0.849, 0.878, 0.838, 0.759, 0.679, 0.672],
    "Depth-Anything": [-0.077, 0.150, 0.020, 0.044, -0.105, -0.084],
    "U-Net using Depth-Anything": [0.059, 0.062, 0.110, 0.101, 0.022, 0.123],
    "U-Net using two images": [0.834, 0.652, 0.880, 0.800, 0.734, 0.619],
}

# Test data
heatmap_data_R2 = {
    "Metric": ["Ratio", "Roundness", "dA", "Area", "dB", "Perimeter"],
    "U-Net": [0.927, 0.603, 0.775, 0.340, 0.570, 0.301],
    "Depth-Anything": [0.699, 0.113, 0.514, 0.530, 0.621, 0.648],
    "U-Net using Depth-Anything": [0.262, -0.175, -0.374, 0.510, -0.100, 0.418],
    "U-Net using two images": [0.873, 0.739, 0.774, 0.394, 0.571, 0.325],
}

# Creating DataFrames
df_heatmap_R = pd.DataFrame(heatmap_data_R).set_index("Metric")
df_heatmap_R2 = pd.DataFrame(heatmap_data_R2).set_index("Metric")

# Creating a single figure with two side-by-side subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

# Function to create heatmap with larger text
def create_heatmap(data, ax, title, cmap, cbar_label):
    sns.heatmap(data, annot=True, fmt=".3f", cmap=cmap, 
                cbar_kws={'label': cbar_label}, ax=ax, 
                annot_kws={'size': 14}, square=True)
    ax.set_title(title, fontsize=18)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=14)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=14)
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)

# Z-score MSE heatmap
create_heatmap(df_heatmap_R, ax1, 'Training correlation Results for Different Models', 'RdYlGn', 'Train correlation')

# Correlation heatmap
create_heatmap(df_heatmap_R2, ax2, 'Test correlation Results for Different Models', 'RdYlGn', 'Test correlation')

# Adjust layout and display
plt.tight_layout()
plt.show()