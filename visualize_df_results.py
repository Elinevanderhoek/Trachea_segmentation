import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

file_path_ground_truth = "...\\df_ground_truth_test.csv"
file_path_prediction = "...\\df_prediction_test.csv"

df_ground_truth = pd.read_csv(file_path_ground_truth)
df_prediction = pd.read_csv(file_path_prediction)

# Example metrics to compare
metrics = ['dA', 'dB', 'Area', 'Perimeter', 'Ratio', 'Roundness']

# Initialize a dictionary to store MSE results
mse_results = {}
z_score_results = {}
correlation_results = {}

# Calculate the MSE for each metric
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

    print(f"Correlation (R) for {metric}: {corr}")
    
    # Bereken de z-score voor de ground truth en prediction (op basis van de ground truth standaarddeviatie)
    mean_ground_truth = aligned_data.iloc[:, 0].mean()
    std_ground_truth = aligned_data.iloc[:, 0].std()

    if std_ground_truth > 0:
        z_score_ground_truth = (aligned_data.iloc[:, 0] - mean_ground_truth) / std_ground_truth
        z_score_prediction = (aligned_data.iloc[:, 1] - mean_ground_truth) / std_ground_truth

        # Bereken de Mean Squared Error tussen de z-scores
        z_mse = mean_squared_error(z_score_ground_truth, z_score_prediction)
        z_score_results[metric] = z_mse

        print(f"Z-score MSE for {metric}: {z_mse}")
    else:
        z_score_results[metric] = np.nan  # Indien standaarddeviatie 0 is, kan de z-score niet berekend worden
        print(f"Z-score MSE for {metric}: cannot be calculated due to zero standard deviation in ground truth")

    # Calculate MSE
    mse = mean_squared_error(aligned_data.iloc[:, 0], aligned_data.iloc[:, 1])
    
    # Store the result in the dictionary
    mse_results[metric] = mse
    print(f"MSE for {metric}: {mse}")

# Sort MSE results
sorted_mse_results = sorted(mse_results.items(), key=lambda x: x[1])
sorted_z_score_results = sorted(z_score_results.items(), key=lambda x: x[1] if not np.isnan(x[1]) else float('inf'))
sorted_correlation_results = sorted(correlation_results.items(), key=lambda x: x[1], reverse=True)

print("\nRanked MSE results:")
for metric, mse in sorted_mse_results:
    print(f"{metric}: {mse}")

print("\nRanked Z-score MSE results:")
for metric, z_mse in sorted_z_score_results:
    print(f"{metric}: {z_mse}")

print("\nRanked Correlation results:")
for metric, corr in sorted_correlation_results:
    print(f"{metric}: {corr}")

# Visualizing the ranked MSE and Z-MSE
fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# Plot for MSE
mse_values_sorted = [x[1] for x in sorted_mse_results]
mse_metrics_sorted = [x[0] for x in sorted_mse_results]
ax[0].barh(mse_metrics_sorted, mse_values_sorted, color='lightgreen')
ax[0].set_xlabel('Mean Squared Error')
ax[0].set_title('Metrics Ranked by MSE')

# Plot for Z-MSE
z_mse_values_sorted = [x[1] for x in sorted_z_score_results]
z_mse_metrics_sorted = [x[0] for x in sorted_z_score_results]
ax[1].barh(z_mse_metrics_sorted, z_mse_values_sorted, color='lightblue')
ax[1].set_xlabel('Z-score Mean Squared Error')
ax[1].set_title('Metrics Ranked by Z-MSE')

plt.tight_layout()
plt.show()

best_mse_metric = min(mse_results, key=mse_results.get)
best_z_mse_metric = min(z_score_results, key=lambda x: z_score_results[x] if not np.isnan(z_score_results[x]) else float('inf'))
best_corr_metric = max(correlation_results, key=correlation_results.get)

print(f"\nMetric with the lowest MSE: {best_mse_metric}")
print(f"Metric with the lowest Z-score MSE: {best_z_mse_metric}")
print(f"Metric with the highest correlation: {best_corr_metric}")

for metric in metrics:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df_ground_truth[metric], y=df_prediction[metric])
    plt.plot([df_ground_truth[metric].min(), df_ground_truth[metric].max()], 
             [df_ground_truth[metric].min(), df_ground_truth[metric].max()], 'r--')
    plt.xlabel('Ground Truth ' + metric.capitalize())
    plt.ylabel('Predicted ' + metric.capitalize())
    plt.title(f'Scatter Plot of {metric.capitalize()}')
    #plt.show()

def bland_altman_plot(ground_truth, prediction, metric):
    avg = (ground_truth + prediction) / 2
    diff = ground_truth - prediction
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)

    plt.figure(figsize=(8, 6))
    plt.scatter(avg, diff, alpha=0.5)
    plt.axhline(mean_diff, color='red', linestyle='--')
    plt.axhline(mean_diff + 1.96 * std_diff, color='blue', linestyle='--')
    plt.axhline(mean_diff - 1.96 * std_diff, color='blue', linestyle='--')
    plt.xlabel('Average of Ground Truth and Prediction')
    plt.ylabel('Difference (Ground Truth - Prediction)')
    plt.title(f'Bland-Altman Plot for {metric.capitalize()}')
    #plt.show()

for metric in metrics:
    bland_altman_plot(df_ground_truth[metric], df_prediction[metric], metric)

for metric in metrics:
    plt.figure(figsize=(8, 6))
    sns.kdeplot(df_ground_truth[metric], label='Ground Truth', shade=True)
    sns.kdeplot(df_prediction[metric], label='Prediction', shade=True)
    plt.xlabel(metric.capitalize())
    plt.ylabel('Density')
    plt.title(f'Distribution of {metric.capitalize()}')
    plt.legend()
    #plt.show()

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
    sns.kdeplot(df_ground_truth[metric], label='Ground Truth', shade=True, ax=axes[i, 2])
    sns.kdeplot(df_prediction[metric], label='Prediction', shade=True, ax=axes[i, 2])
    axes[i, 2].set_xlabel(metric.capitalize())
    axes[i, 2].set_ylabel('Density')
    axes[i, 2].set_title(f'Distribution of {metric.capitalize()}')
    axes[i, 2].legend()

plt.tight_layout()
#plt.show()
