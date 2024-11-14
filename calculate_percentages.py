
## PCA cross-validation prediction model and roundness cross-validation prediction model
## Eline van der Hoek
## 14-11-2024

## Input: file containing transformed dataset
    # Transformed datset should look like this:
    # ID            Area    Shortest diameter   Longest diameter    Roundness   Ratio diameter  Rest lumen%
    # img_exp       xxx     xxx                 xxx                 xxx         xxx             xxx
    # img2_exp      xxx     xxx                 xxx                 xxx         xxx             xxx                    
    # img_insp      xxx     xxx                 xxx                 xxx         xxx             xxx            
    # img2_insp     xxx     xxx                 xxx                 xxx         xxx             xxx 
    # ...           xxx     xxx                 xxx                 xxx         xxx             xxx      

    # Rest lumen% is the estimated percentage of rest lumen by the physician (= 100% - %collapse)
## Output: 
    # Principal Component 1: xxx
    # Principal Component 2: xxx

    # Cross-Validated PCA Model R²: xxx
    # Cross-Validated PCA Model MAE: xxx
    # Cross-Validated PCA Model MSE: xxx

    # Cross-Validated Roundness Model R²: xxx
    # Cross-Validated Roundness Model MAE: xxx
    # Cross-Validated Roundness Model MSE: xxx

    # Scatter plot (actual vs predicted % remaining lumen)
    # Bland-Altman plot (Average of Actual and Predicted Values vs Difference between Predicted and Actual)

# PCA cv prediction model
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the original data
# Replace file path with your file path
file_path = "C:\\Users\\ehoek3\\OneDrive - UMC Utrecht\\Tracheomalacie\\transformed_dataset.xlsx"
df = pd.read_excel(file_path)

# 2. Prepare data for PCA (exclude 'ID' and the target column '% Rest lumen')
X = df.drop(columns=['ID', '% Rest lumen'])
y = df['% Rest lumen']  # Target variable

# 3. Standardize the features for PCA
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# 4. Perform PCA, keeping 95% of the variance
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_standardized)

# Explained variance ratio of each principal component
explained_variance = pca.explained_variance_ratio_
print("\nExplained Variance by Principal Components:")
for i, variance in enumerate(explained_variance):
    print(f"Principal Component {i + 1}: {variance:.2f}")

# 5. Cross-validation setup (Linear Regression with 5-fold cross-validation)
model = LinearRegression()

# Perform cross-validation (predict on each fold's test set)
y_pred = cross_val_predict(model, X_pca, y, cv=5)

# Evaluate cross-validated model
r2_cv = r2_score(y, y_pred)
mae_cv = mean_absolute_error(y, y_pred)
mse_cv = mean_squared_error(y, y_pred)

# Print the results
print(f'\nCross-Validated PCA Model R²: {r2_cv:.2f}')
print(f'Cross-Validated PCA Model MAE: {mae_cv:.2f}')
print(f'Cross-Validated PCA Model MSE: {mse_cv:.2f}')

# 6. Plot actual vs predicted values for cross-validation
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, color='blue', label='Predicted vs Actual')

# Add the ideal line (y = x)
plt.plot([min(y), max(y)], [min(y), max(y)], color='green', linestyle='--', label=f"Ideal line (y=x) (MAE = {mae_cv:.2f})")

# Fit a regression line through the cross-validated data
line_model = LinearRegression()
line_model.fit(y.values.reshape(-1, 1), y_pred)
y_line = line_model.predict(np.array([min(y), max(y)]).reshape(-1, 1))

# Plot the fitted regression line through the points
plt.plot([min(y), max(y)], y_line, color='red', linestyle='-', label='Regression line through points')

plt.title(f'Cross-validated PCA model: actual vs predicted % remaining lumen (R² = {r2_cv:.2f})')
plt.xlabel('Actual % remaining lumen')
plt.ylabel('Predicted % remaining lumen')
plt.legend()
plt.grid(True)
plt.show()

# 7. Bland-Altman plot
# Calculate the differences and averages
differences = y_pred - y
averages = (y_pred + y) / 2

# Calculate the mean and standard deviation of the differences
mean_diff = np.mean(differences)
std_diff = np.std(differences)

# Plot the Bland-Altman plot
plt.figure(figsize=(8, 6))
plt.scatter(averages, differences, color='blue', label='Differences vs Averages')

# Add the mean difference (center line)
plt.axhline(mean_diff, color='red', linestyle='--', label=f'Mean Difference = {mean_diff:.2f}')

# Add the limits of agreement (mean ± 1.96*std)
plt.axhline(mean_diff + 1.96*std_diff, color='green', linestyle='--', label=f'LOA + 1.96*std = {mean_diff + 1.96*std_diff:.2f}')
plt.axhline(mean_diff - 1.96*std_diff, color='green', linestyle='--', label=f'LOA - 1.96*std = {mean_diff - 1.96*std_diff:.2f}')

# Labels and title
plt.title('Bland-Altman Plot PCA model: Actual vs Predicted % Remaining Lumen')
plt.xlabel('Average of Actual and Predicted Values')
plt.ylabel('Difference between Predicted and Actual')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

# ________________________________________________________
# Roundness cv prediction model
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the original data
# Replace file path with your file path
file_path = "C:\\Users\\ehoek3\\OneDrive - UMC Utrecht\\Tracheomalacie\\transformed_dataset.xlsx"
df = pd.read_excel(file_path)

# 2. Prepare data for the model (use only 'roundness' as the feature)
X = df[['Roundness']]  # Feature
y = df['% Rest lumen']  # Target variable

# 3. Cross-validation setup (Linear Regression with 5-fold cross-validation)
model = LinearRegression()

# Perform cross-validation (predict on each fold's test set)
y_pred_cv = cross_val_predict(model, X, y, cv=5)

# Evaluate cross-validated model
r2_cv = r2_score(y, y_pred_cv)
mae_cv = mean_absolute_error(y, y_pred_cv)
mse_cv = mean_squared_error(y, y_pred_cv)

# Print the results
print(f'\nCross-Validated Roundness Model R²: {r2_cv:.2f}')
print(f'Cross-Validated Roundness Model MAE: {mae_cv:.2f}')
print(f'Cross-Validated Roundness Model MSE: {mse_cv:.2f}')

# 4. Plot actual vs predicted values for cross-validation
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred_cv, color='blue', label='Predicted vs Actual')

# Add the ideal line (y = x)
plt.plot([min(y), max(y)], [min(y), max(y)], color='green', linestyle='--', label=f"Ideal line (y=x) (MAE = {mae_cv:.2f})")

# Fit a regression line through the cross-validated points
line_model = LinearRegression()
line_model.fit(y.values.reshape(-1, 1), y_pred_cv)
y_line = line_model.predict(np.array([min(y), max(y)]).reshape(-1, 1))

# Plot the fitted regression line through the points
plt.plot([min(y), max(y)], y_line, color='red', linestyle='--', label='Regression line through points')

plt.title(f'Cross-validated roundness model: actual vs predicted % remaining lumen (R² = {r2_cv:.2f})')
plt.xlabel('Actual % remaining lumen')
plt.ylabel('Predicted % remaining lumen')
plt.legend()
plt.grid(True)
plt.show()

# 7. Bland-Altman plot
# Calculate the differences and averages
differences = y_pred_cv - y
averages = (y_pred_cv + y) / 2

# Calculate the mean and standard deviation of the differences
mean_diff = np.mean(differences)
std_diff = np.std(differences)

# Plot the Bland-Altman plot
plt.figure(figsize=(8, 6))
plt.scatter(averages, differences, color='blue', label='Differences vs Averages')

# Add the mean difference (center line)
plt.axhline(mean_diff, color='red', linestyle='--', label=f'Mean Difference = {mean_diff:.2f}')

# Add the limits of agreement (mean ± 1.96*std)
plt.axhline(mean_diff + 1.96*std_diff, color='green', linestyle='--', label=f'LOA + 1.96*std = {mean_diff + 1.96*std_diff:.2f}')
plt.axhline(mean_diff - 1.96*std_diff, color='green', linestyle='--', label=f'LOA - 1.96*std = {mean_diff - 1.96*std_diff:.2f}')

# Labels and title
plt.title('Bland-Altman Plot roundness model: Actual vs Predicted % Remaining Lumen')
plt.xlabel('Average of Actual and Predicted Values')
plt.ylabel('Difference between Predicted and Actual')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()