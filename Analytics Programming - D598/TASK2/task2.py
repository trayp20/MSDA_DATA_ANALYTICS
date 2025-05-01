import pandas as pd

# Step 1: Import the data file into a DataFrame
df = pd.read_excel("D598 Data Set.xlsx")

# Step 2: Identify duplicate rows in the dataset
duplicates = df[df.duplicated()]
print("Duplicate Rows Found:")
print(duplicates)

# Step 3: Remove duplicate rows
df = df.drop_duplicates()
print("\nDataFrame after removing duplicates:")
print(df)

# Step 4: Group all IDs by state and run descriptive statistics
grouped_stats = df.groupby("Business State").agg({
    'Total Revenue': ['mean', 'median', 'min', 'max'],
    'Total Long-term Debt': ['mean', 'median', 'min', 'max'],
    'Debt to Equity': ['mean', 'median', 'min', 'max']
})
grouped_stats.columns = ['_'.join(col) for col in grouped_stats.columns]
grouped_stats.reset_index(inplace=True)
print("\nGrouped Descriptive Statistics:")
print(grouped_stats)

# Step 5: Filter businesses with negative debt-to-equity ratios
negative_debt_equity = df[df['Debt to Equity'] < 0]
print("\nBusinesses with Negative Debt-to-Equity Ratios:")
print(negative_debt_equity)

# Step 6: Create a new DataFrame for debt-to-income ratios
df['Debt-to-Income Ratio'] = df['Total Long-term Debt'] / df['Total Revenue']
print("\nDebt-to-Income Ratio DataFrame:")
print(df[['Business ID', 'Debt-to-Income Ratio']])

# Step 7: Concatenate the new DataFrame with the original data
df_combined = pd.concat([df, df[['Debt-to-Income Ratio']]], axis=1)
print("\nCombined DataFrame:")
print(df_combined)
