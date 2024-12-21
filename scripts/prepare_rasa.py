import re
import json
import pandas as pd
from pathlib import Path

# Initialize empty DataFrames for train and test data
train_df = pd.DataFrame()
test_df = pd.DataFrame()

# Recursively search for metadata_train.json and metadata_test.json
current_dir = Path('/home/tts/ttsteam/datasets/rasa')
metadata_files = list(current_dir.rglob('metadata_*.json'))

def clean_text(text):
    if text is not None:
        text = re.sub(r"\(.*?\)", "", text)
        text = re.sub(r"\[.*?\]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = text.replace('"', '')
        text = text.replace('|', '।')
    return text


for metadata_file in metadata_files:
  # Determine if the file is train or test based on its name
  data_type = 'train' if 'train' in metadata_file.name else 'test'

  # Load the metadata json into a pandas DataFrame
  data = json.load(open(metadata_file))[data_type]
  df = pd.DataFrame(data)

  # Get the name of the 2nd level parent folder
  second_level_parent = metadata_file.parents[1].name

  # Add the parent folder name as a prefix to the gender column
  df['gender'] = second_level_parent + "_" + df['gender']

  # Find the corresponding 'wavs' folder
  wavs_folder = metadata_file.parent / 'wavs'

  # Update file paths
  df['filepath'] = df.apply(lambda row: str(wavs_folder / f"{row['filename']}.wav"), axis=1)
  df['text'] = df['text'].apply(clean_text)

  # Append rows to the corresponding DataFrame
  if data_type == 'train':
    train_df = pd.concat([train_df, df], ignore_index=True)
  else:
    test_df = pd.concat([test_df, df], ignore_index=True)

# Convert style and gender to categorical codes (optional)
train_df['style'] = pd.Categorical(train_df['style']).codes
train_df['gender'] = pd.Categorical(train_df['gender']).codes

test_df['style'] = pd.Categorical(test_df['style']).codes
test_df['gender'] = pd.Categorical(test_df['gender']).codes

# Save train and test data to CSV files
for data_type, df in [("train", train_df), ("test", test_df)]:
  output_csv_path = current_dir / f"{data_type}_metadata.csv"
  # Save to CSV in the prescribed format
  df[['filepath', 'text', 'gender', 'style']].to_csv(
      output_csv_path, sep='|', index=False, header=False
  )
  print(f"{data_type.capitalize()} metadata CSV saved to: {output_csv_path}")