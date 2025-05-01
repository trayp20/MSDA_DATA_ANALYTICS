import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load the dataset
df = pd.read_csv("/Users/trayvoniouspendleton/IdeaProjects/d600-statistical-data-mining/task1/D600 Task 1 Dataset 1 Housing Information.csv")

# Part C
# Select the relevant colums
dependent_variable ="Price"
independent_variables = ["CrimeRate", "SchoolRating"]
df_selected = df[[dependent_variable] + independent_variables]

# Generate and display descriptive statistics
df_descriptive_stats = df_selected.describe()
print(df_descriptive_stats)

# Create histograms for each variable
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

df_selected.hist(column=dependent_variable, bins=30, ax=axes[0])
axes[0].set_title("Distribution of House Prices")
axes[0].set_xlabel("Price")
axes[0].set_ylabel("Frequency")

df_selected.hist(column=independent_variables[0], bins=30, ax=axes[1])
axes[1].set_title("Distribution of Crime Rate")
axes[1].set_xlabel("Crime Rate")
axes[1].set_ylabel("Frequency")

df_selected.hist(column=independent_variables[1], bins=30, ax=axes[2])
axes[2].set_title("Distribution of School Rating")
axes[2].set_xlabel("School Rating")
axes[2].set_ylabel("Frequency")

plt.tight_layout()
plt.show()

# Create scatter plots to visualize relationships
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Scatter plot: Crime Rate vs. House Price
axes[0].scatter(df[independent_variables[0]], df[dependent_variable], alpha=0.5)
axes[0].set_xlabel("Crime Rate")
axes[0].set_ylabel("House Price")
axes[0].set_title("Crime Rate vs House Price")

# Scatter plot: School Rating vs. House Price
axes[1].scatter(df[independent_variables[1]], df[dependent_variable], alpha=0.5)
axes[1].set_xlabel("School Rating")
axes[1].set_ylabel("House Price")
axes[1].set_title("School Rating vs House Price")

plt.tight_layout()
plt.show()

# Part D
#Part D1 Splitting Dataset
from sklearn.model_selection import train_test_split

# Define independent and dependent variables
X = df[["CrimeRate", "SchoolRating"]]  # Independent variables
y = df["Price"]  # Dependent variable

# Split data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save training and testing datasets
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

# Save to CSV files
train_data.to_csv("training_data.csv", index=False)
test_data.to_csv("testing_data.csv", index=False)

print("Data split into training and testing sets and saved as CSV files.")

# Part D2
# Add constant term (intercept) for regression
X_train_const = sm.add_constant(X_train)

# Step 1: Fit initial regression model
model = sm.OLS(y_train, X_train_const).fit()

# Step 2: Check if optimization is needed (remove variables with p-value > 0.05)
p_values = model.pvalues.drop("const")  # Exclude the intercept
if any(p_values > 0.05):
    print("Optimization needed: Removing insignificant variables...\n")

    while True:
        max_p = p_values.max()  # Find the highest p-value
        if max_p > 0.05:
            worst_feature = p_values.idxmax()
            print(f"Removing {worst_feature} with p-value {max_p:.4f}")
            X_train_const = X_train_const.drop(columns=[worst_feature])  # Drop variable
            model = sm.OLS(y_train, X_train_const).fit()  # Refit model
            p_values = model.pvalues.drop("const")  # Recalculate p-values
        else:
            break  # Stop when all p-values < 0.05

# Print optimized model summary
print(model.summary())

# Extract required values
r_squared = model.rsquared
adjusted_r_squared = model.rsquared_adj
f_statistic = model.fvalue
coefficients = model.params
p_values = model.pvalues

# Print required outputs for submission
print("\nOptimized Model Performance Metrics (Training Data):")
print(f"R-Squared: {r_squared:.4f}")
print(f"Adjusted R-Squared: {adjusted_r_squared:.4f}")
print(f"F-Statistic: {f_statistic:.4f}")

print("\nOptimized Coefficient Estimates:")
print(coefficients)

print("\nOptimized P-Values of Independent Variables:")
print(p_values)

# Part D3
from sklearn.metrics import mean_squared_error

# Generate predictions on the training dataset using the optimized model
y_pred_train = model.predict(X_train_const)

# Compute Mean Squared Error (MSE)
mse_train = mean_squared_error(y_train, y_pred_train)

# Print required output for submission
print("\nModel Performance Metrics (Training Data):")
print(f"Mean Squared Error (MSE) on Training Set: {mse_train:.2f}")

# D4
# Add a constant term to the test set
X_test_const = sm.add_constant(X_test)

# Drop any features that were removed during D2 optimization
X_test_const = X_test_const[X_train_const.columns]

# Generate predictions on the test dataset using the optimized model
y_pred_test = model.predict(X_test_const)

# Compute Mean Squared Error (MSE) on test data
mse_test = mean_squared_error(y_test, y_pred_test)

# Print required output for submission
print("\nModel Performance Metrics (Test Data):")
print(f"Mean Squared Error (MSE) on Test Set: {mse_test:.2f}")


