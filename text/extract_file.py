import tarfile
from tqdm import tqdm
import os

language = 'mar'
zip_file_name = 'mar_f_m_all_48k'

# Paths
input_tar_gz_path = f'/nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/ai4bharat_internal/{language}/{zip_file_name}.tar.gz'
output_directory_path = '/nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/pilot_rasa/audio/'

# Open the tar.gz file
with tarfile.open(input_tar_gz_path, 'r:gz') as tar:
    # Get the total number of members
    total_members = len(tar.getmembers())

    # Extract the tar.gz file with progress bar
    with tqdm(total=total_members, desc='Extracting', unit='file') as pbar:
        for member in tar.getmembers():
            tar.extract(member, path=output_directory_path)
            pbar.update(1)

print("Extraction completed.")

# mv('/nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/pilot_rasa/audio_48k/audio/mar_f_m_all_48k/', '/nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/pilot_rasa/audio/')
# rename('/nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/pilot_rasa/audio/mar_f_m_all_48k', '/nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/pilot_rasa/audio/mar')

# Move and rename directories
src_directory = os.path.join(output_directory_path, 'audio_48k/audio/{zip_file_name}')
dst_directory = os.path.join(output_directory_path, '{language}')

# Check if the source directory exists
if os.path.exists(src_directory):
    # Move and rename the directory
    shutil.move(src_directory, dst_directory)
    print(f"Moved and renamed directory from {src_directory} to {dst_directory}")
else:
    print(f"Source directory {src_directory} does not exist.")