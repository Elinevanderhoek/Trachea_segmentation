
## Correlation results by comparing df_ground_truth and df_prediction
## Eline van der Hoek
## 14-11-2024

## Input: df_ground_truth and df_prediction
    # The file path must be adapted so that the right model (and training/test) output is used 
## Output: 
    # Ranked Correlation results:
    # Ratio:        xxx
    # dA:           xxx
    # Roundness:    xxx
    # dB:           xxx
    # Area:         xxx
    # Perimeter:    xxx

    # Metric with the highest correlation: xxx

    # Plot showing 1) scatter plots, 2) bland-altman plots and 3) distribution plots of all metrics

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

file_path_ground_truth = "C:\\Users\\ehoek3\\OneDrive - UMC Utrecht\\Tracheomalacie\\Python\\Depth-Anything-V2\\Output_unet_depth_combi\\Test\\df_ground_truth_test.csv"
file_path_prediction = "C:\\Users\\ehoek3\\OneDrive - UMC Utrecht\\Tracheomalacie\\Python\\Depth-Anything-V2\\Output_unet_depth_combi\\Test\\df_prediction_test.csv"

df_ground_truth = pd.read_csv(file_path_ground_truth)
df_prediction = pd.read_csv(file_path_prediction)

# Example metrics to compare
metrics = ['dA', 'dB', 'Area', 'Perimeter', 'Ratio', 'Roundness']

# Initialize a dictionary to store correlation results
correlation_results = {}

# Calculate the correlation for each metric
for metric in metrics:
    # Drop any NaN values from either ground truth or prediction
    ground_truth_valid = df_ground_truth[metric].dropna()
    prediction_valid = df_prediction[metric].dropna()

    # Align indexes after dropping NaN values
    aligned_data = pd.concat([ground_truth_valid, prediction_valid], axis=1, join='inner')
    aligned_data = aligned_data.replace([np.inf, -np.inf], np.nan).dropna()

    # Calculate correlation (R)
    corr, _ = pearsonr(aligned_data.iloc[:, 0], aligned_data.iloc[:, 1])
    correlation_results[metric] = corr

sorted_correlation_results = sorted(correlation_results.items(), key=lambda x: x[1], reverse=True)

print("\nRanked Correlation results:")
for metric, corr in sorted_correlation_results:
    print(f"{metric}: {corr}")
best_corr_metric = max(correlation_results, key=correlation_results.get)

print(f"\nMetric with the highest correlation: {best_corr_metric}")

fig, axes = plt.subplots(len(metrics), 3, figsize=(18, 5 * len(metrics)))

for i, metric in enumerate(metrics):
    # Scatter Plot
    sns.scatterplot(x=df_ground_truth[metric], y=df_prediction[metric], ax=axes[i, 0])

    # Adding the x = y line
    min_value = min(df_ground_truth[metric].min(), df_prediction[metric].min())
    max_value = max(df_ground_truth[metric].max(), df_prediction[metric].max())
    axes[i, 0].plot([min_value, max_value], [min_value, max_value], 'r--')

    axes[i, 0].set_xlabel('Ground Truth ' + metric.capitalize())
    axes[i, 0].set_ylabel('Predicted ' + metric.capitalize())
    axes[i, 0].set_title(f'Scatter Plot of {metric.capitalize()}')

    # Bland-Altman Plot
    avg = (df_ground_truth[metric] + df_prediction[metric]) / 2
    diff = df_ground_truth[metric] - df_prediction[metric]
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    axes[i, 1].scatter(avg, diff, alpha=0.5)
    axes[i, 1].axhline(mean_diff, color='red', linestyle='--')
    axes[i, 1].axhline(mean_diff + 1.96 * std_diff, color='blue', linestyle='--')
    axes[i, 1].axhline(mean_diff - 1.96 * std_diff, color='blue', linestyle='--')
    axes[i, 1].set_xlabel('Average of Ground Truth and Prediction')
    axes[i, 1].set_ylabel('Difference')
    axes[i, 1].set_title(f'Bland-Altman Plot for {metric.capitalize()}')

    # Distribution Plot
    sns.kdeplot(df_ground_truth[metric], label='Ground Truth', fill=True, ax=axes[i, 2])
    sns.kdeplot(df_prediction[metric], label='Prediction', fill=True, ax=axes[i, 2])
    axes[i, 2].set_xlabel(metric.capitalize())
    axes[i, 2].set_ylabel('Density')
    axes[i, 2].set_title(f'Distribution of {metric.capitalize()}')
    axes[i, 2].legend()

plt.tight_layout()
plt.show()