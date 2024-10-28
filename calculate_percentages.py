import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error


file_path = "...\\transformed_dataset.xlsx"  # Update with your file path
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
