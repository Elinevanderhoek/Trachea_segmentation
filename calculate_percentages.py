# import pandas as pd
# import joblib
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# # Load your original data
# data = pd.read_excel("C:\\Users\\ehoek3\\OneDrive - UMC Utrecht\\Tracheomalacie\\transformed_dataset.xlsx")  # Adjust file name
# X = data.drop(columns=['% Rest lumen', 'ID'])  # Features
# y = data['% Rest lumen']  # Target variable

# # Standardize the data
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# correlation = data.corr()['% Collapse'].drop('% Collapse')

# # Display correlation results
# print("\nCorrelation with % Collapse:")
# print(correlation)

# # Sort correlations
# sorted_correlation = correlation.sort_values(ascending=False)
# print("\nSorted Correlation with % Collapse:")
# print(sorted_correlation)

# # Perform PCA
# pca = PCA(n_components=0.95)  # Keep 95% variance, adjust as needed
# X_pca = pca.fit_transform(X_scaled)

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# # Build and train a regression model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Make predictions
# y_pred = model.predict(X_test)

# # Evaluate the model
# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f'MAE: {mae}, MSE: {mse}, R²: {r2}')

# # Save the model for future use
# joblib.dump(model, 'collapse_prediction_model.pkl')

# # Load and use the model for new data
# new_data = pd.read_excel('C:\\Users\\ehoek3\\OneDrive - UMC Utrecht\\Tracheomalacie\\transformed_dataset_test.xlsx')
# # Exclude 'ID' from new data
# new_data_numeric = new_data.drop(columns=['ID', '% Rest lumen'])
# new_data_scaled = scaler.transform(new_data_numeric)  # Standardize new data
# new_data_pca = pca.transform(new_data_scaled)  # Apply PCA
# predicted_collapse = model.predict(new_data_pca)  # Predict % Collapse

# # Add predictions to DataFrame
# new_data['Predicted % Collapse'] = predicted_collapse
# new_data.to_excel('C:\\Users\\ehoek3\\OneDrive - UMC Utrecht\\Tracheomalacie\\new_data_with_predictions.xlsx', index=False)

# __________________________________________________________________________________

# import pandas as pd

# # Load the original dataset (replace the path with your actual file path)
# file_path = "C:\\Users\\ehoek3\\OneDrive - UMC Utrecht\\Tracheomalacie\\resultaten_5test.xlsx"
# df = pd.read_excel(file_path)

# # Split the dataset into expiratory and inspiratory datasets
# expiratory_columns = ['ID', 'Area (expiratoir)', 'Perimeter (expiratoir)', 'Shortest Diameter (expiratoir)',
#                       'Longest Diameter (expiratoir)', 'Roundness (expiratoir)', 'Ratio Diameters (expiratoir)',
#                       '% Rest lumen (expiratoir)']  # Add more columns as needed

# inspiratory_columns = ['ID', 'Area (inspiratoir)', 'Perimeter (inspiratoir)', 'Shortest Diameter (inspiratoir)',
#                        'Longest Diameter (inspiratoir)', 'Roundness (inspiratoir)', 'Ratio Diameters (inspiratoir)',
#                        '% Rest lumen (inspiratoir)']  # Add more columns as needed

# # Create separate dataframes for expiratory and inspiratory
# exp_df = df[expiratory_columns].copy()
# insp_df = df[inspiratory_columns].copy()

# # Append 'exp' or 'insp' to the ID column
# exp_df['ID'] = exp_df['ID'].astype(str) + '_exp'
# insp_df['ID'] = insp_df['ID'].astype(str) + '_insp'

# # Rename columns for consistency
# exp_df.columns = ['ID', 'Area', 'Perimeter', 'Shortest Diameter', 'Longest Diameter', 'Roundness', 'Ratio Diameters', '% Collapse']
# insp_df.columns = ['ID', 'Area', 'Perimeter', 'Shortest Diameter', 'Longest Diameter', 'Roundness', 'Ratio Diameters', '% Collapse']

# # Concatenate expiratory and inspiratory rows
# combined_df = pd.concat([exp_df, insp_df])

# # Reset index for neatness
# combined_df.reset_index(drop=True, inplace=True)

# # Save to a new Excel file
# combined_df.to_excel("C:\\Users\\ehoek3\\OneDrive - UMC Utrecht\\Tracheomalacie\\transformed_dataset_test.xlsx", index=False)

# print("Data has been reshaped and saved to 'transformed_dataset.xlsx'")

# _________________________________________________________________________________

# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import matplotlib.pyplot as plt

# # 1. Load the original data
# file_path = "C:\\Users\\ehoek3\\OneDrive - UMC Utrecht\\Tracheomalacie\\transformed_dataset.xlsx"
# df = pd.read_excel(file_path)

# # 2. Prepare data for PCA (exclude 'ID' and the target column '% Rest lumen')
# X = df.drop(columns=['ID', '% Rest lumen'])
# y = df['% Rest lumen']  # Target variable

# # 3. Standardize the features for PCA
# scaler = StandardScaler()
# X_standardized = scaler.fit_transform(X)

# # 4. Perform PCA, keeping 95% of the variance
# pca = PCA(n_components=0.95)
# X_pca = pca.fit_transform(X_standardized)

# # Explained variance ratio of each principal component
# explained_variance = pca.explained_variance_ratio_
# print("\nExplained Variance by Principal Components:")
# for i, variance in enumerate(explained_variance):
#     print(f"Principal Component {i + 1}: {variance:.2f}")

# # 5. Cross-validation setup (Linear Regression with 5-fold cross-validation)
# model = LinearRegression()

# # Perform cross-validation (predict on each fold's test set)
# y_pred = cross_val_predict(model, X_pca, y, cv=5)

# # Evaluate cross-validated model
# r2_cv = r2_score(y, y_pred)
# mae_cv = mean_absolute_error(y, y_pred)
# mse_cv = mean_squared_error(y, y_pred)

# #Print the results
# print(f'\nCross-Validated PCA Model R²: {r2_cv:.2f}')
# print(f'Cross-Validated PCA Model MAE: {mae_cv:.2f}')
# print(f'Cross-Validated PCA Model MSE: {mse_cv:.2f}')

# # 6. Plot actual vs predicted values for cross-validation
# plt.figure(figsize=(8, 6))
# plt.scatter(y, y_pred_cv, color='blue', label='Predicted vs Actual')
# plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--', label=f"Ideal line (y=x) (MAE = {mae_cv:.2f})")
# plt.title(f'Cross-validated PCA model: actual vs predicted % remaining lumen (R² = {r2_cv:.2f})')
# plt.xlabel(f'Actual % remaining lumen')
# plt.ylabel(f'Predicted % remaining Lumen')
# plt.legend()
# plt.grid(True)
# plt.show()

# Save new data with predictions to an Excel file
# new_data.to_excel('C:\\Users\\ehoek3\\OneDrive - UMC Utrecht\\Tracheomalacie\\new_data_with_predictions.xlsx', index=False)
# print("\nNew data with predictions saved to 'new_data_with_predictions.xlsx'")

# ________________________________________________________
# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import cross_val_predict
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import matplotlib.pyplot as plt

# # 1. Load the original data
# file_path = "C:\\Users\\ehoek3\\OneDrive - UMC Utrecht\\Tracheomalacie\\transformed_dataset.xlsx"
# df = pd.read_excel(file_path)

# # 2. Prepare data for the model (use only 'roundness' as the feature)
# X = df[['Roundness']]  # Feature
# y = df['% Rest lumen']  # Target variable

# # 3. Cross-validation setup (Linear Regression with 5-fold cross-validation)
# model = LinearRegression()

# # Perform cross-validation (predict on each fold's test set)
# y_pred_cv = cross_val_predict(model, X, y, cv=5)

# # Evaluate cross-validated model
# r2_cv = r2_score(y, y_pred_cv)
# mae_cv = mean_absolute_error(y, y_pred_cv)
# mse_cv = mean_squared_error(y, y_pred_cv)

# # Print the results
# print(f'\nCross-Validated Roundness Model R²: {r2_cv:.2f}')
# print(f'Cross-Validated Roundness Model MAE: {mae_cv:.2f}')
# print(f'Cross-Validated Roundness Model MSE: {mse_cv:.2f}')

# # 4. Plot actual vs predicted values for cross-validation
# plt.figure(figsize=(8, 6))
# plt.scatter(y, y_pred_cv, color='blue', label='Predicted vs Actual')
# plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--', label=f"Ideal line (y=x) (MAE = {mae_cv:.2f})")
# plt.title(f'Cross-validated roundness model: actual vs predicted % remaining lumen (R² = {r2_cv:.2f})')
# plt.xlabel(f'Actual % remaining Lumen')
# plt.ylabel(f'Predicted % remaining Lumen')
# plt.legend()
# plt.grid(True)
# plt.show()
# _________________________________________________________

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error


file_path = "C:\\Users\\ehoek3\\OneDrive - UMC Utrecht\\Tracheomalacie\\transformed_dataset.xlsx"  # Update with your file path
df = pd.read_excel(file_path)
df = df.drop(columns=['ID'])  # Features

# Calculate correlation of all metrics with '% Rest lumen'
correlation = df.corr()['% Rest lumen'].drop('% Rest lumen')

# Get the 3 most correlated metrics (highest absolute correlation values)
top_3_metrics = correlation.abs().nlargest(3).index

models = {}

# Visualize the top 3 metrics with scatter plots and regression line
for metric in top_3_metrics:
    plt.figure(figsize=(8, 6))
    
    # Scatter plot of the metric vs % Rest lumen
    plt.scatter(df[metric], df['% Rest lumen'], color='blue', label=f'{metric} vs % Rest Lumen')
    
    # Fit a linear regression model to get the best-fit line
    X = df[[metric]].values.reshape(-1, 1)  # Reshape for sklearn compatibility
    y = df['% Rest lumen'].values
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict the values using the model
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    models[metric] = model

    mae = mean_absolute_error(y, y_pred)

    # Plot the regression line (best-fit line)
    plt.plot(df[metric], y_pred, color='red', linestyle='--', label=f"Regression line (MAE = {mae:.2f})")
    
    # Adding titles and labels
    plt.title(f'Scatter plot: {metric} vs % remaining lumen (R² = {r2:.2f})')
    plt.xlabel(f'{metric}')
    plt.ylabel('% Remaining lumen')
    plt.grid(True)
    plt.legend()
    
    # Show the plot
    plt.show()

# ____________________________________________________________

# import pandas as pd

# # Assuming you have the data loaded into a DataFrame (df)
# data = {
#     '% Rest lumen': [50, 70, 75, 40, 5, 15, 80, 75, 60, 20],
#     'Predicted % Roundness': [45, 84, 72, 65, 4, 60, 93, 83, 72, 66],
#     'Difference (Roundness)': [5, -14, 3, 25, -1, 45, 13, 8, 12, 46],
#     'Predicted % Ratio Diameters': [53, 69, 82, 54, 19, 57, 80, 92, 64, 54],
#     'Difference (Ratio Diameters)': [48, -1, 7, 14, 14, 42, 0, 17, 4, 34],
#     'Predicted % Shortest Diameter': [64, 86, 76, 40, 27, 71, 93, 82, 67, 41],
#     'Difference (Shortest Diameter)': [59, 16, 1, 0, 22, 56, 13, 7, 7, 21]
# }

# # Create the dataframe
# df = pd.DataFrame(data)

# # Calculate basic statistics for each column
# statistics = df.describe()

# # Calculate correlations
# correlations = df.corr()

# # Calculate R² score for each prediction column with respect to % Rest lumen
# from sklearn.metrics import r2_score

# r2_scores = {
#     'Roundness': r2_score(df['% Rest lumen'], df['Predicted % Roundness']),
#     'Ratio Diameters': r2_score(df['% Rest lumen'], df['Predicted % Ratio Diameters']),
#     'Shortest Diameter': r2_score(df['% Rest lumen'], df['Predicted % Shortest Diameter'])
# }

# # Calculate Mean Absolute Error (MAE) for each prediction column
# mae = {
#     'Roundness': (df['Difference (Roundness)'].abs()).mean(),
#     'Ratio Diameters': (df['Difference (Ratio Diameters)'].abs()).mean(),
#     'Shortest Diameter': (df['Difference (Shortest Diameter)'].abs()).mean()
# }

# # Print the results
# print("Basic Statistics:")
# print(statistics)

# print("\nCorrelations:")
# print(correlations)

# print("\nR² Scores for each model:")
# for metric, r2 in r2_scores.items():
#     print(f"{metric}: {r2:.2f}")

# print("\nMean Absolute Error (MAE) for each model:")
# for metric, error in mae.items():
#     print(f"{metric}: {error:.2f}")

# _______________________________________________________

# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt
# from sklearn.metrics import r2_score, mean_absolute_error

# # 1. Load the original data
# file_path = "C:\\Users\\ehoek3\\OneDrive - UMC Utrecht\\Tracheomalacie\\transformed_dataset.xlsx"
# df = pd.read_excel(file_path)

# # 2. Prepare data for PCA (exclude 'ID' and the target column '% Rest lumen')
# X = df.drop(columns=['ID', '% Rest lumen'])
# y = df['% Rest lumen']  # Target variable

# # 3. Standardize the features for PCA
# scaler = StandardScaler()
# X_standardized = scaler.fit_transform(X)

# # 4. Perform PCA
# pca = PCA()
# X_pca = pca.fit_transform(X_standardized)

# # 5. Create scatter plot for each principal component
# num_components = min(2, X_pca.shape[1])  # Plot up to 2 components or less if there are fewer

# for i in range(num_components):
#     # Prepare data for the current principal component
#     pc_values = X_pca[:, i]
    
#     # Fit a linear regression line
#     model = LinearRegression()
#     pc_values_reshaped = pc_values.reshape(-1, 1)
#     model.fit(pc_values_reshaped, y)
    
#     # Predict using only this principal component
#     y_pred = model.predict(pc_values_reshaped)
    
#     # Calculate metrics
#     r2 = r2_score(y, y_pred)
#     mae = mean_absolute_error(y, y_pred)
    
#     # Create the scatter plot
#     plt.figure(figsize=(10, 6))
#     plt.scatter(pc_values, y, color='blue', alpha=0.6, label=f"PC{i+1} vs % remaining lumen")
#     plt.plot(pc_values, y_pred, color='red', linewidth=2, label=f"Regression line (MAE = {mae:.2f})")
#     plt.legend()
    
#     plt.title(f'Principal component {i+1} vs % remaining lumen (R² = {r2:.2f})')
#     plt.xlabel(f'Principal component {i+1}')
#     plt.ylabel('% Remaining lumen')
    
#     plt.grid(True)
#     plt.show()

# # 6. Print explained variance ratio
# explained_variance = pca.explained_variance_ratio_
# print("\nExplained Variance Ratio by Principal Components:")
# for i, variance in enumerate(explained_variance[:num_components]):
#     print(f"Principal Component {i + 1}: {variance:.4f}")

# # 7. Create a DataFrame with PC values and save to Excel
# pc_df = pd.DataFrame(X_pca[:, :num_components], columns=[f'PC{i+1}' for i in range(num_components)])
# pc_df['% Rest lumen'] = y
# output_file = "principal_components_output.xlsx"
# pc_df.to_excel(output_file, index=False)

# print(f"\nPrincipal components saved to '{output_file}'")