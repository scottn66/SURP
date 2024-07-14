import os
import pandas as pd
import random

def process_data_pipeline():
    # Step 1: Combine Batches
    data_dir = "../dat/batch_results"
    additional_file = "../dat/basic_dataframe_with_content.csv"

    data_frames = [pd.read_csv(os.path.join(data_dir, filename)) for filename in os.listdir(data_dir) if filename.endswith(".csv")]
    merged_df = pd.concat(data_frames, ignore_index=True)
    print(f"Shape of merged_df: {merged_df.shape}")

    additional_df = pd.read_csv(additional_file)[['url', 'headline', 'content']]
    final_df = pd.merge(merged_df, additional_df, left_on="Answer.url", right_on="url", how="left")
    print(f"Final_df shape: {final_df.shape}, '\n and final columns', {final_df.columns} \n\n\n")

    duplicate_count = final_df.duplicated().sum()
    duplicate_rows = final_df[final_df.duplicated(keep=False)]
    common_columns = set.intersection(*[set(df.columns) for df in data_frames])
    uncommon_columns = set.union(*[set(df.columns) for df in data_frames]) - common_columns
    overlapping_columns_info = [(col, final_df[col].dtype) for col in common_columns if col in final_df.columns]

    print(f"Overlapping columns with datatypes: {overlapping_columns_info}")
    print(final_df.head())
    print(final_df.describe())
    print('Unique urls: ', len(final_df['url'].unique()))

    # Step 2: Clean Data
    print("Initial data shape:", final_df.shape)
    print(final_df.head())

    missing_values = final_df.isnull().sum()
    missing_percentage = (missing_values / len(final_df)) * 100

    columns_to_drop = missing_percentage[missing_percentage > 50.0].index
    data_cleaned = final_df.drop(columns=columns_to_drop)

    for column in data_cleaned.columns:
        if data_cleaned[column].dtype in ['float64', 'int64']:
            data_cleaned[column].fillna(data_cleaned[column].mean(), inplace=True)
        elif data_cleaned[column].dtype == 'object':
            data_cleaned[column].fillna(data_cleaned[column].mode()[0], inplace=True)

    data_cleaned.drop_duplicates(inplace=True)

    print(f"\nData shape after cleaning: {data_cleaned.shape}")
    print(data_cleaned.head())

    # Step 3: Preprocess Data
    columns_to_remove = [
        'Description', 'Keywords', 'Reward', 'AssignmentDurationInSeconds', 
        'AutoApprovalDelayInSeconds', 'AssignmentStatus', 
        'Last30DaysApprovalRate', 'Last7DaysApprovalRate',
        'HITTypeId', 'RequesterAnnotation', 'Expiration', 'MaxAssignments', 
        'CreationTime', 'HITId', 'LifetimeApprovalRate', 'AutoApprovalTime', 
        'SubmitTime', 'AcceptTime'
    ]
    data_cleaned.drop(columns=columns_to_remove, inplace=True)

    random_index = random.randint(0, len(data_cleaned) - 1)
    random_example = data_cleaned.iloc[random_index]
    print(f"Random example (index {random_index}):\n{random_example}")

    # Step 4: Model Training
    data_cleaned['Answer.language1'] = data_cleaned['Answer.language1'].apply(lambda x: 'English' if x == 'English' else 'non-English')
    data_cleaned['Answer.country'] = data_cleaned['Answer.country'].apply(lambda x: 'US' if x == 'United States' else 'non-US')

    random_index = random.randint(0, len(data_cleaned) - 1)
    random_row = data_cleaned.iloc[random_index]

    def output_styled(row):
        demographics = [row['Answer.age'], row['Answer.gender'], row['Answer.country'], row['Answer.language1'], row['Answer.politics']]
        social_media = [row[col] for col in ['WorkTimeInSeconds', 'Answer.facebook-hours', 'Answer.instagram-hours', 'Answer.reddit-hours', 'Answer.twitter-hours']]
        article_information = [row['Answer.newsOutlet'], row['url'], row['headline'], row['content']]
        X = demographics + social_media + article_information
        y = row['Answer.bias-question']
        return X, y
    data_pairs = [output_styled(row) for _, row in data_cleaned.iterrows()]
    df = pd.DataFrame(data_pairs, columns=['X', 'y'])
    df.to_csv('../dat/DSPy_data.csv', index=False)
    print(df)

if __name__ == "__main__":
    process_data_pipeline()
