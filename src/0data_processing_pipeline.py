import os
import pandas as pd
import random

def load_data_frames(data_dir):
    data_frames = []
    all_columns = set()
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):
            file_path = os.path.join(data_dir, filename)
            df = pd.read_csv(file_path)
            data_frames.append(df)
            all_columns.update(df.columns)
    return data_frames, all_columns

def merge_data_frames(data_frames):
    return pd.concat(data_frames, ignore_index=True)

def load_additional_data(additional_file):
    additional_df = pd.read_csv(additional_file)
    return additional_df[['url', 'headline', 'content']]

def merge_with_additional(merged_df, additional_df):
    return pd.merge(merged_df, additional_df, left_on="Answer.url", right_on="url", how="left")

def get_common_and_uncommon_columns(data_frames, all_columns):
    common_columns = set(data_frames[0].columns)
    for df in data_frames[1:]:
        common_columns.intersection_update(df.columns)
    uncommon_columns = all_columns - common_columns
    return common_columns, uncommon_columns

def handle_missing_values(data):
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
    for column in data_cleaned.select_dtypes(include=['float64', 'int64']).columns:
        data_cleaned[column].fillna(data_cleaned[column].mean(), inplace=True)

    for column in data_cleaned.select_dtypes(include=['object']).columns:
        data_cleaned[column].fillna(data_cleaned[column].mode()[0], inplace=True)

    return data_cleaned

def perform_eda(data_cleaned):
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

def preprocess_data(data):
    columns_to_remove = [
        'Description', 'Keywords', 'Reward', 'AssignmentDurationInSeconds', 
        'AutoApprovalDelayInSeconds', 'AssignmentStatus', 
        'Last30DaysApprovalRate', 'Last7DaysApprovalRate',
        'HITTypeId', 'RequesterAnnotation', 'Expiration', 'MaxAssignments', 
        'CreationTime', 'HITId', 'LifetimeApprovalRate', 'AutoApprovalTime', 
        'SubmitTime', 'AcceptTime'
    ]
    data.drop(columns=columns_to_remove, inplace=True)

    random_index = random.randint(0, len(data) - 1)
    random_example = data.iloc[random_index]
    print(f"Random example (index {random_index}):\n{random_example}")

    return data

def data_formatting(data):
    data['Answer.language1'] = data['Answer.language1'].apply(lambda x: 'English' if x == 'English' else 'non-English')
    data['Answer.country'] = data['Answer.country'].apply(lambda x: 'US' if x == 'United States' else 'non-US')

    def output_styled(row):
        demographics = [row['Answer.age'], row['Answer.gender'], row['Answer.country'], row['Answer.language1'], row['Answer.politics']]
        social_media = [row[col] for col in ['WorkTimeInSeconds', 'Answer.facebook-hours', 'Answer.instagram-hours', 'Answer.reddit-hours', 'Answer.twitter-hours']]
        article_information = [row['Answer.newsOutlet'], row['url'], row['headline'], row['content']]
        X = demographics + social_media + article_information
        y = row['Answer.bias-question']
        return X, y

    data_pairs = [output_styled(row) for _, row in data.iterrows()]
    df = pd.DataFrame(data_pairs, columns=['X', 'y'])
    df.to_csv('../dat/DSPy_data.csv', index=False)
    print(df)

def main():
    data_dir = "../dat/batch_results"
    additional_file = "../dat/article_text_contents.csv"
    output_file = "../dat/merged_df.csv"

    try:
        data_frames, all_columns = load_data_frames(data_dir)
        merged_df = merge_data_frames(data_frames)
        print(f"Shape of merged_df: {merged_df.shape}")

        additional_df = load_additional_data(additional_file)
        final_df = merge_with_additional(merged_df, additional_df)
        print(f"Final_df shape: {final_df.shape}, '\n and final columns', {final_df.columns} \n\n\n")

        duplicate_count = final_df.duplicated().sum()
        duplicate_rows = final_df[final_df.duplicated(keep=False)]

        common_columns, uncommon_columns = get_common_and_uncommon_columns(data_frames, all_columns)
        overlapping_columns_info = [(col, final_df[col].dtype) for col in common_columns if col in final_df.columns]

        print(f"Overlapping columns with datatypes: {overlapping_columns_info}")
        print(final_df.head())
        print(final_df.describe())
        print('Unique urls: ', len(final_df['url'].unique()))

        data_cleaned = handle_missing_values(final_df)
        data_cleaned = data_cleaned.drop_duplicates()
        print("\nData shape after removing duplicates:", data_cleaned.shape)
        
        # Save cleaned data to a new CSV file
        data_cleaned.to_csv("../dat/cleaned_df.csv", index=False)

        perform_eda(data_cleaned)
        
        # Save cleaned and analyzed data to a new CSV file (optional)
        data_cleaned.to_csv("../dat/cleaned_analyzed_df.csv", index=False)

        preprocessed_df = preprocess_data(data_cleaned)
        data_formatting(preprocessed_df)
        
        print("Data processing pipeline completed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
