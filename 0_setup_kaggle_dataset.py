
import os
import shutil
import kagglehub

# Download latest version
path = kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset")

# Create 'data' directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Move all CSV files from the dataset path to the 'data' directory
for filename in os.listdir(path):
    if filename.endswith('.csv'):
        full_file_path = os.path.join(path, filename)
        shutil.copy(full_file_path, 'data')

print("CSV files saved to 'data/' directory.")
