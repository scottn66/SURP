import pandas as pd
import random
import numpy as np

"""
Objective: Prepare processed data for DSPy notebook

needs
	(1) Only relevant variables (ie have predictive power)
	(2) input output tuple
		a.) (all features, is-biased)

"""

# Load the cleaned DataFrame
data = pd.read_csv("../dat/cleaned_data_processed.csv")

# Simplify language and country columns
data['Answer.language1'] = data['Answer.language1'].apply(lambda x: 'English' if x == 'English' else 'non-English')
data['Answer.country'] = data['Answer.country'].apply(lambda x: 'US' if x == 'United States' else 'non-US')


random_index = random.randint(0, len(data) - 1)
random_row = data.iloc[random_index]

def output_styled(random_row=random_row):
	# for col_value in random_row:
	# 	print('\n', col_value)
	# 	print('\t -->', random_row[col_value])
	age = random_row['Answer.age']
	country = random_row['Answer.country']
	gender = random_row['Answer.gender']
	language = random_row['Answer.language1']
	politics = random_row['Answer.politics']
	news_outlet = random_row['Answer.newsOutlet']
	work_time_seconds = random_row['WorkTimeInSeconds']
	article_number = random_row['Answer.articleNumber']
	facebook_hours = random_row['Answer.facebook-hours']
	instagram_hours = random_row['Answer.instagram-hours']
	reddit_hours = random_row['Answer.reddit-hours']
	twitter_hours = random_row['Answer.twitter-hours']
	url = random_row['url']
	headline = random_row['headline']
	content = random_row['content']
	bias_question = random_row['Answer.bias-question']

	demographics = [age, gender, country, language, politics]
	social_media = [("work_time_seconds", work_time_seconds), 
					("facebook_hours", facebook_hours), 
					("instagram_hours", instagram_hours),
					("reddit_hours", reddit_hours),
					("twitter_hours", twitter_hours)]
	article_information = [news_outlet, url, headline, content]

	X = demographics + social_media + article_information
	y = bias_question

	pair = (X, y)

	return pair



output = output_styled()

# Prepare data for CSV
data_for_csv = {
    'X': [output[0]],  # List of all features and values
    'y': [output[1]]   # Bias question value
}

# Create DataFrame
df = pd.DataFrame(data_for_csv)

# Save DataFrame to CSV
df.to_csv('../dat/DSPy_data.csv', index=False)

print(df)