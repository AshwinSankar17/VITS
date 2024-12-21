import os
import pandas as pd

# Paths
input_csv_path = '/nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/pilot_rasa/csv_file/mar_f_m_all.csv'
output_directory = '/nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/mar/'
audio_path_prefix = '/nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/pilot_rasa/audio/mar'

# Read the CSV file with the correct delimiter
df = pd.read_csv(input_csv_path, delimiter=',')

# Modify the 'data_id' column
df['data_id'] = df['data_id'].apply(lambda x: f"{audio_path_prefix}/{x}.wav")

# Remove rows where the audio file does not exist
df = df[df['data_id'].apply(os.path.exists)]

# Replace 'Female' with 1 and 'Male' with 0 in the 'speaker' column
df['speaker'] = df['speaker'].replace({'Female': 1, 'Male': 0})

# Remove newline characters from the 'transcript' column
df['transcript'] = df['transcript'].str.replace('\n', ' ')

# Keep only the columns 'data_id', 'transcript', and 'speaker'
df = df[['data_id', 'transcript', 'speaker', 'style']]

emotions_mapping = {
        'SANGRAH': 0,
        'WIKI': 0,
        'INDIC':0,
        'PROPER NOUN':0,
        'HAPPY' : 1,
        'SAD':2,
        'ANGER' :3,
        'FEAR': 4,
        'SURPRISE':5,
        'DISGUST':6,
        'ALEXA':7,
        'DIGI':8,
        'BOOK':9,
        'CONV':10,
        'NEWS':11,
        'UMANG':12,
        'BB':13,
    }

df['style'] = df['style'].map(emotions_mapping)

num_rows = len(df)

if num_rows < 100:
    raise ValueError(f"Not enough data to sample 100 rows. Available rows: {num_rows}")

# Randomly sample 100 rows for the test set
df_test = df.sample(n=100, random_state=42)

# Use the remaining rows for the train set
df_train = df.drop(df_test.index)

# Save the train and test DataFrames to new CSV files using '|' as a separator
df_train.to_csv(f'{output_directory}/metadata_train_vits_raw.csv', index=False, sep='|', header=None)
df_test.to_csv(f'{output_directory}/metadata_test_vits_raw.csv', index=False, sep='|', header=None)

print("CSV files have been split, modified, and saved.")
