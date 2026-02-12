import pandas as pd

# 1. Load the csv file
df = pd.read_csv('train.csv')

# 2. Apply the filter
# We check three conditions at the same time using '&'
filtered_data = df[
    (df['task'] == 'Sentiment') & 
    (df['variety'] == 'en-IN') & 
    (df['source'] == 'Reddit')
]

# 3. Save the result to a new file
filtered_data.to_csv('train_IN_Reddit_Sentiment.csv', index=False)

print(f"Found {len(filtered_data)} rows matching your criteria.")