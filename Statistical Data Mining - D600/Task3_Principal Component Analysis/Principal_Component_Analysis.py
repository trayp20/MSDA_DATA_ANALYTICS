import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv("/Users/trayvoniouspendleton/IdeaProjects/d600-statistical-data-mining/task_3/D600 Task 3 Dataset 1 Housing Information.csv")

# Part D2
# Selecting relevant continuous numerical variables
continuous_vars = [
    "Price", "SquareFootage", "NumBathrooms", "NumBedrooms",
    "BackyardSpace", "CrimeRate", "SchoolRating", "AgeOfHome",
    "DistanceToCityCenter", "EmploymentRate", "PropertyTaxRate",
    "RenovationQuality", "LocalAmenities", "TransportAccess",
    "PreviousSalePrice"
]

# Standardizing the data
scaler = StandardScaler()
df[continuous_vars] = scaler.fit_transform(df[continuous_vars])

# Save the standardized dataset
df.to_csv("D600_Task3_Standardized_Housing.csv", index=False)

# Display a preview of the standardized dataset
print(df.head())

# Part D3
# Generate descriptive statistics
desc_stats = df[continuous_vars].describe().T  # Transpose for better readability

# Display results
print(desc_stats)

# Part E1
# Apply PCA
pca = PCA()
principal_components = pca.fit_transform(df[continuous_vars])

# Convert principal components into a DataFrame
pca_df = pd.DataFrame(principal_components, columns=[f"PC{i+1}" for i in range(len(continuous_vars))])

# Save the matrix of principal components
pca_df.to_csv("D600_Task3_Principal_Components.csv", index=False)

# Display the matrix
print("Matrix of Principal Components:")
print(pca_df.head())

# Part E2
# Variance explained by each component
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Generate a Scree Plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--', label="Individual Variance")
plt.plot(range(1, len(explained_variance) + 1), cumulative_variance, marker='s', linestyle='-', label="Cumulative Variance")
plt.axhline(y=0.85, color='r', linestyle='--', label="85% Variance Explained")
plt.xlabel("Number of Principal Components")
plt.ylabel("Explained Variance")
plt.title("Scree Plot: Principal Component Analysis")
plt.legend()
plt.grid()
plt.show()

# Apply Elbow Rule: Find the point where variance gain slows
elbow_point = np.argmax(np.diff(explained_variance) < 0.01) + 1

# Apply Kaiser Rule: Keep components with variance > 1/N
kaiser_threshold = 1 / len(continuous_vars)
kaiser_components = np.sum(explained_variance > kaiser_threshold)

print(f"Elbow Rule suggests retaining {elbow_point} components.")
print(f"Kaiser Rule suggests retaining {kaiser_components} components.")

# Part E3
# Create a DataFrame to display the variance explained by each principal component
variance_df = pd.DataFrame({
    "Principal Component": [f"PC{i+1}" for i in range(len(explained_variance))],
    "Variance Explained": explained_variance
})

# Save the variance explained data to a CSV file
variance_df.to_csv("D600_Task3_PCA_Variance_Explained.csv", index=False)

# Display the variance explained
print("\nVariance Explained by Each Principal Component:")
print(variance_df)

# Part E4
# Display total variance explained by all components
total_variance_explained = np.sum(explained_variance)
print(f"\nTotal Variance Explained by All Principal Components: {total_variance_explained:.4f}")

# Identify the number of components needed to explain 85% of variance
num_components_85 = np.argmax(np.cumsum(explained_variance) >= 0.85) + 1
print(f"Number of Principal Components to Retain (85% Variance Explained): {num_components_85}")

# Display the cumulative variance explained by the selected number of components
cumulative_variance = np.cumsum(explained_variance)
variance_summary = pd.DataFrame({
    "Principal Component": [f"PC{i+1}" for i in range(len(explained_variance))],
    "Cumulative Variance Explained": cumulative_variance
})

# Display summary results
print("\nPCA Summary:")
print(variance_summary.head(num_components_85))

# Part F1
# Use the selected principal components for regression
num_components_to_keep = num_components_85  # Use the number identified in Part E4
selected_features = [f"PC{i+1}" for i in range(num_components_to_keep)]

# Create the final dataset with selected PCs and target variable (Price)
pca_df["Price"] = df["Price"]  # Add the target variable back to the PCA-transformed dataset
final_dataset = pca_df[selected_features + ["Price"]]  # Keep only selected PCs + Price

# Split into 80% training and 20% testing datasets
train_data, test_data = train_test_split(final_dataset, test_size=0.2, random_state=42)

# Save training and testing datasets as two separate CSV files
train_data.to_csv("D600_Task3_Training_Dataset.csv", index=False)
test_data.to_csv("D600_Task3_Test_Dataset.csv", index=False)

# Display dataset sizes
print(f"Training dataset size: {train_data.shape[0]} samples")
print(f"Test dataset size: {test_data.shape[0]} samples")

# Part F2
# Load the training dataset
train_df = pd.read_csv("D600_Task3_Training_Dataset.csv")

# Split into features (X) and target variable (y)
X_train = train_df.drop(columns=["Price"])  # Features (Principal Components)
y_train = train_df["Price"]  # Target variable (House Price)

# Add a constant term for intercept in regression
X_train = sm.add_constant(X_train)

# Fit the initial regression model
model = sm.OLS(y_train, X_train).fit()

# Display initial regression summary
print("\nInitial Regression Model Summary:")
print(model.summary())

# Perform Backward Stepwise Selection to optimize the model
def backward_elimination(X, y, significance_level=0.05):
    """
    Performs backward elimination to remove features with high p-values.
    Stops when all remaining features have p-values below the given significance level.
    """
    X = sm.add_constant(X)  # Ensure constant term is present
    while True:
        model = sm.OLS(y, X).fit()
        p_values = model.pvalues[1:]  # Ignore constant term
        max_p_value = p_values.max()
        if max_p_value > significance_level:
            feature_to_remove = p_values.idxmax()
            print(f"Dropping '{feature_to_remove}' (p={max_p_value:.5f})")
            X = X.drop(columns=[feature_to_remove])
        else:
            break
    return X, model

# Optimize the model using backward elimination
X_train_optimized, optimized_model = backward_elimination(X_train, y_train)

# Extract required regression metrics
regression_results = {
    "R-squared": optimized_model.rsquared,
    "Adjusted R-squared": optimized_model.rsquared_adj,
    "F-statistic": optimized_model.fvalue,
    "Probability F-statistic": optimized_model.f_pvalue,
}

# Extract coefficient estimates and p-values
coefficients = optimized_model.params
p_values = optimized_model.pvalues

# Save the results to a text file
with open("D600_Task3_Regression_Summary.txt", "w") as file:
    file.write("Optimized Regression Model Summary:\n")
    file.write(str(optimized_model.summary()))
    file.write("\n\nKey Regression Metrics:\n")
    for key, value in regression_results.items():
        file.write(f"{key}: {value:.4f}\n")
    file.write("\nCoefficient Estimates:\n")
    for coef, value in coefficients.items():
        file.write(f"{coef}: {value:.4f}\n")
    file.write("\nP-values of Independent Variables:\n")
    for var, p_val in p_values.items():
        file.write(f"{var}: {p_val:.6f}\n")

# Save the optimized training dataset
X_train_optimized.to_csv("D600_Task3_X_Train_Optimized.csv", index=False)

# Display key metrics in console
print("\nKey Regression Metrics:")
for key, value in regression_results.items():
    print(f"{key}: {value:.4f}")

print("\nCoefficient Estimates:")
print(coefficients)

print("\nP-values of Independent Variables:")
print(p_values)

print("\nModel Optimization Completed. Summary saved to 'D600_Task3_Regression_Summary.txt'.")

# Part F3
# Load optimized training dataset and target variable
X_train_optimized = pd.read_csv("D600_Task3_X_Train_Optimized.csv")
y_train = pd.read_csv("D600_Task3_Training_Dataset.csv")["Price"]

# Train the optimized regression model
optimized_model = sm.OLS(y_train, sm.add_constant(X_train_optimized)).fit()

# Compute Mean Squared Error (MSE) on training data
train_mse = mean_squared_error(y_train, optimized_model.predict(sm.add_constant(X_train_optimized)))

# Save MSE to file
with open("D600_Task3_Training_MSE.txt", "w") as file:
    file.write(f"Training Set Mean Squared Error (MSE): {train_mse:.4f}\n")

# Display MSE result
print(f"Training Set Mean Squared Error (MSE): {train_mse:.4f}")

# Part F4
# Load the optimized training dataset (features selected in D2)
X_train_optimized = pd.read_csv("D600_Task3_X_Train_Optimized.csv")

# Load the full test dataset
test_df = pd.read_csv("D600_Task3_Test_Dataset.csv")

# Ensure the test dataset only contains the variables from the optimized model (D2)
selected_features = X_train_optimized.columns.tolist()  # Get feature names (excluding "const")
if "const" in selected_features:
    selected_features.remove("const")  # Remove "const" since it is added separately

# Select the matching columns from test dataset
X_test = test_df[selected_features]  # Match test set to optimized training features
y_test = test_df["Price"]  # Target variable

# Add a constant term to match the training dataset
X_test = sm.add_constant(X_test)

# Load the target variable from the training dataset
train_df = pd.read_csv("D600_Task3_Training_Dataset.csv")
y_train = train_df["Price"]

# Train the optimized regression model (from F2)
optimized_model = sm.OLS(y_train, sm.add_constant(X_train_optimized)).fit()

# Make predictions on the test dataset
y_test_pred = optimized_model.predict(X_test)

# Compute Mean Squared Error (MSE) on test dataset
test_mse = mean_squared_error(y_test, y_test_pred)

# Save the test MSE result to a text file
with open("D600_Task3_Test_MSE.txt", "w") as file:
    file.write(f"Mean Squared Error (MSE) on Test Set: {test_mse:.4f}\n")

# Display the MSE result
print(f"\nMean Squared Error (MSE) on Test Set: {test_mse:.4f}")