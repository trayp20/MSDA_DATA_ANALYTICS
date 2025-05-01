import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, accuracy_score

# Load the dataset
df = pd.read_csv('/Users/trayvoniouspendleton/IdeaProjects/d600-statistical-data-mining/task_2/D600 Task 2 Dataset 1 Housing Information.csv')
#Part C2
#Generate descriptive statistics for selected variables
descriptive_stats = df[['IsLuxury', 'CrimeRate', 'SchoolRating']].describe()

# Count the number of luxury vs. non-luxury homes
luxury_counts = df['IsLuxury'].value_counts()

# Display the descriptive statistics
print("Descriptive Statistics:\n", descriptive_stats)

# Display the class distribution for IsLuxury
print("\nLuxury Home Count:\n", luxury_counts)

#Part C3
import matplotlib.pyplot as plt

# Create histograms for CrimeRate and SchoolRating
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Histogram for CrimeRate
ax[0].hist(df['CrimeRate'], bins=30, edgecolor='black', alpha=0.7)
ax[0].set_title('Distribution of Crime Rate')
ax[0].set_xlabel('Crime Rate')
ax[0].set_ylabel('Frequency')

# Histogram for SchoolRating
ax[1].hist(df['SchoolRating'], bins=30, edgecolor='black', alpha=0.7, color='orange')
ax[1].set_title('Distribution of School Rating')
ax[1].set_xlabel('School Rating')
ax[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# Boxplots comparing CrimeRate and SchoolRating with IsLuxury
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Boxplot for CrimeRate vs. IsLuxury
df.boxplot(column='CrimeRate', by='IsLuxury', ax=ax[0])
ax[0].set_title('Crime Rate by Luxury Status')
ax[0].set_xlabel('Luxury Status (0 = Not Luxury, 1 = Luxury)')
ax[0].set_ylabel('Crime Rate')

# Boxplot for SchoolRating vs. IsLuxury
df.boxplot(column='SchoolRating', by='IsLuxury', ax=ax[1])
ax[1].set_title('School Rating by Luxury Status')
ax[1].set_xlabel('Luxury Status (0 = Not Luxury, 1 = Luxury)')
ax[1].set_ylabel('School Rating')

plt.tight_layout()
plt.show()

# Part D1
# Select the relevant columns for the logistic regression model
X = df[['CrimeRate', 'SchoolRating']]  # Independent variables
y = df['IsLuxury']  # Dependent variable

# Split the dataset into training (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Combine features and target variable for saving
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# Save to CSV
train_df.to_csv("housing_train.csv", index=False)
test_df.to_csv("housing_test.csv", index=False)

# Display the size of the training and test datasets
print("Training set size:", len(X_train))
print("Test set size:", len(X_test))

#Part D2
# Add a constant to the independent variables
X_train_const = sm.add_constant(X_train)

# Fit logistic regression model
logit_model = sm.Logit(y_train, X_train_const)
result = logit_model.fit()

# Perform backward stepwise elimination
p_values_initial = result.pvalues

# Remove 'CrimeRate' if its p-value is greater than 0.05
if p_values_initial['CrimeRate'] > 0.05:
    X_train_optimized = X_train.drop(columns=['CrimeRate'])
else:
    X_train_optimized = X_train  # Keep original if all variables are significant

# Add a constant to the optimized independent variables
X_train_optimized_const = sm.add_constant(X_train_optimized)

# Fit the optimized logistic regression model
logit_model_optimized = sm.Logit(y_train, X_train_optimized_const)
result_optimized = logit_model_optimized.fit()

# Extract optimized model parameters
aic_optimized = result_optimized.aic
bic_optimized = result_optimized.bic
pseudo_r2_optimized = result_optimized.prsquared
coefficients_optimized = result_optimized.params
p_values_optimized = result_optimized.pvalues

# Display optimized model summary
print(result_optimized.summary())

# Print extracted values
print("\nOptimized Model Metrics:")
print(f"AIC: {aic_optimized}")
print(f"BIC: {bic_optimized}")
print(f"Pseudo R-squared: {pseudo_r2_optimized}")
print("\nOptimized Coefficient Estimates:")
print(coefficients_optimized)
print("\nOptimized P-values:")
print(p_values_optimized)

# D3
# Ensure X_test_optimized matches X_train_optimized
if 'CrimeRate' in p_values_initial and p_values_initial['CrimeRate'] > 0.05:
    X_test_optimized = X_test.drop(columns=['CrimeRate'])
else:
    X_test_optimized = X_test

# Add a constant to the optimized test dataset
X_test_optimized_const = sm.add_constant(X_test_optimized)

# Make predictions on the test set using the optimized model
y_pred_probs = result_optimized.predict(X_test_optimized_const)
y_pred = (y_pred_probs >= 0.5).astype(int)  # Convert probabilities to binary predictions

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Display results
print("Confusion Matrix:\n", conf_matrix)
print("\nAccuracy Score:", accuracy)

# Part D4
# Evaluate model performance on the test dataset
conf_matrix_test = confusion_matrix(y_test, y_pred)
accuracy_test = accuracy_score(y_test, y_pred)

# Display results
print("Confusion Matrix (Test Set):\n", conf_matrix_test)
print("\nAccuracy Score (Test Set):", accuracy_test)


