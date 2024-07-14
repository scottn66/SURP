import pandas as pd
import random

# Load the cleaned DataFrame
data = pd.read_csv("../dat/cleaned_df.csv")

# Feature Engineering: Removing specified columns
columns_to_remove = [
    'Description', 'Keywords', 'Reward', 'AssignmentDurationInSeconds', 
    'AutoApprovalDelayInSeconds', 'AssignmentStatus', 
    'Last30DaysApprovalRate', 'Last7DaysApprovalRate'
]
data.drop(columns=columns_to_remove, inplace=True)

secondary_columns_to_remove = [
    'HITTypeId', 'RequesterAnnotation', 'Expiration', 'MaxAssignments', 
    'CreationTime', 'HITId', 'LifetimeApprovalRate', 'AutoApprovalTime', 
    'SubmitTime', 'AcceptTime'
]
data.drop(columns=secondary_columns_to_remove, inplace=True)

# Convert necessary columns to appropriate types if required
# For example: numeric_columns = ['WorkTimeInSeconds', 'Answer.articleNumber', 'Answer.batch', 'Answer.facebook-hours', 'Answer.instagram-hours']
# for col in numeric_columns:
#     data[col] = pd.to_numeric(data[col], errors='coerce')
# print(data)

# column: total unique values
# col_uniqueness = {}
# for col in list(data.columns):
# 	total_unique_values = len(data[col].unique())
# 	# print(type(data[col][0]), num_unique_values)
# 	# if num_unique_values == 1:
# 	# 	print(f"get rid of {col}")
# 	col_uniqueness[col] = total_unique_values
# 	#print(col, ': ', type(data[col][0]), data[col].unique())

# sorted_dict = dict(sorted(col_uniqueness.items(), key=lambda item: item[1]))
# for key, value in sorted_dict.items():
# 	print(value, key, type(data[key][0]))
# 	if value > 100:
# 		print('\t\ttoo many unique values')
# 	else:
# 		print('\t', list(data[key].unique()))

"""
Columns with only one value: 
[Description, Keywords, Reward, AssignmentDurationInSeconds, AutoApprovalDelayInSeconds, AssignmentStatus, Last30DaysApprovalRate, Last7DaysApprovalRate]

"""
# DATA TYPE CONVERSIONS
# numeric_columns = ['WorkTimeInSeconds', 'Answer.articleNumber', 'Answer.batch', 'Answer.facebook-hours', 'Answer.instagram-hours']
# print('\n\n Date data')
# for col in numeric_columns:
# 	print(data[col][0])
# 	data[col] = pd.to_numeric(data[col], errors='coerce')

# label_encoders = {}
# for column in data.select_dtypes(include=['object']).columns:
# 	label_encoders[column] = LabelEncoder()
# 	data[column] = label_encoders[column].fit_transform(data[column].astype(str))

# Print data types to verify conversions
# print(data.dtypes)


# Select a random example
num_examples = len(data)
random_index = random.randint(0, num_examples - 1)
random_example = data.iloc[random_index]

# Print each value for the random example
print(f"Random example (index {random_index}):\n")
for col, val in random_example.items():
    print(f"{col}: {val}")

# Save the cleaned DataFrame to a new CSV file in the ../dat/ folder
output_path = "../dat/cleaned_data_processed.csv"
data.to_csv(output_path, index=False)

print(f"\nData saved to {output_path}")
