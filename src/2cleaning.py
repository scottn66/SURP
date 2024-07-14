import pandas as pd

# Load the merged DataFrame
data = pd.read_csv("../dat/merged_df.csv")

# Check the initial state of the data
print("Initial data shape:", data.shape)
print(data.head())

# Check for missing values
missing_values = data.isnull().sum()
print("\nMissing values per column:")
print(missing_values)

# Percentage of missing values
missing_percentage = (missing_values / len(data)) * 100
print("\nPercentage of missing values per column:")
print(missing_percentage)

# Define a threshold for dropping columns (e.g., drop columns with more than 50% missing values)
threshold = 50.0
columns_to_drop = missing_percentage[missing_percentage > threshold].index
print("\nColumns to drop (more than 50% missing values):")
print(columns_to_drop)

# Drop columns with more than 50% missing values
data_cleaned = data.drop(columns=columns_to_drop)
print("\nData shape after dropping columns with too many missing values:", data_cleaned.shape)

# Handle remaining missing values
# Option 1: Drop rows with any remaining missing values
# data_cleaned = data_cleaned.dropna()

# Option 2: Fill missing values with a specific value (e.g., mean for numerical columns, mode for categorical columns)
for column in data_cleaned.select_dtypes(include=['float64', 'int64']).columns:
    data_cleaned[column].fillna(data_cleaned[column].mean(), inplace=True)

for column in data_cleaned.select_dtypes(include=['object']).columns:
    data_cleaned[column].fillna(data_cleaned[column].mode()[0], inplace=True)

# Check the state after handling missing values
print("\nData shape after handling missing values:", data_cleaned.shape)
print(data_cleaned.head())

# Remove duplicates
data_cleaned = data_cleaned.drop_duplicates()
print("\nData shape after removing duplicates:", data_cleaned.shape)

# Save cleaned data to a new CSV file
data_cleaned.to_csv("../dat/cleaned_df.csv", index=False)

# 2. Basic Exploratory Data Analysis (EDA)
# Print basic information about the DataFrame
print("Basic Information:")
print(data_cleaned.info())

# Print summary statistics
print("\nSummary Statistics:")
print(data_cleaned.describe())

# Print the first few rows of the DataFrame
print("\nFirst Few Rows:")
print(data_cleaned.head())

# Count the number of unique values in each column
print("\nUnique Values in Each Column:")
print(data_cleaned.nunique())

# Analyze the target variable distribution (replace 'target_variable' with your actual target column)
target_column = 'Answer.bias-question'  # Replace with your actual target column
if target_column in data_cleaned.columns:
    print(f"\nDistribution of {target_column}:")
    print(data_cleaned[target_column].value_counts())
else:
    print(f"\nTarget column '{target_column}' not found in the cleaned data.")

# Save cleaned and analyzed data to a new CSV file (optional)
data_cleaned.to_csv("../dat/cleaned_analyzed_df.csv", index=False)

print("Data cleaning and analysis completed.")
