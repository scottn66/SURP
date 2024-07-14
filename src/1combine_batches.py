import os
import pandas as pd

# Directory containing the batch result files
data_dir = "../dat/batch_results"
additional_file = "../dat/basic_dataframe_with_content.csv"
output_file = "../dat/merged_df.csv"

# List to hold DataFrames
data_frames = []
all_columns = set()

# Load each file into a DataFrame and append to the list
for filename in os.listdir(data_dir):
    if filename.endswith(".csv"):  # Adjust the file extension as necessary
        file_path = os.path.join(data_dir, filename)
        df = pd.read_csv(file_path)
        data_frames.append(df)
        all_columns.update(df.columns)

# Merge all DataFrames
merged_df = pd.concat(data_frames, ignore_index=True)
print(f"shape of merged_df: {merged_df.shape}")

# Load the additional CSV file
additional_df = pd.read_csv(additional_file)

# Select only the necessary columns from additional_df
additional_df = additional_df[['url', 'headline', 'content']]
# print(additional_df)

# Merge with the additional DataFrame on 'Answer.url' column from merged_df and 'url' column from additional_df
final_df = pd.merge(merged_df, additional_df, left_on="Answer.url", right_on="url", how="left")

print(f"final_df shape: {final_df.shape}, '\n and final columns', {final_df.columns} \n\n\n")

# Count duplicates in the final merged DataFrame
duplicate_count = final_df.duplicated().sum()

# Get all duplicate rows, including first occurrences
duplicate_rows = final_df[final_df.duplicated(keep=False)]

# Find common columns in the initial merged DataFrame
common_columns = set(data_frames[0].columns)
for df in data_frames[1:]:
    common_columns.intersection_update(df.columns)

# Find uncommon columns
uncommon_columns = all_columns - common_columns

# Get column names and data types of overlapping columns in the final DataFrame
overlapping_columns_info = [(col, final_df[col].dtype) for col in common_columns if col in final_df.columns]

# # Print results
# print(f"Number of duplicates: {duplicate_count}")
# print(f"Common columns: {common_columns}")
# print(f"Uncommon columns: {uncommon_columns}")
print(f"Overlapping columns with datatypes: {overlapping_columns_info}")

# # Display the first few rows of the final merged DataFrame
print(final_df.head())
print(final_df.describe())
print('unique urls: ', len(final_df['url'].unique()))

# Save the final DataFrame to a CSV file
final_df.to_csv(output_file, index=False)

print(f"The merged DataFrame has been saved to {output_file}")
