import os
# from setuptools import setup
import gdown

def download_file():
    files_dir = os.path.join(os.path.dirname(__file__), 'files')
    if not os.path.exists(files_dir):
        os.makedirs(files_dir)

    url = 'https://drive.google.com/uc?id=16HCaqszdyeXFOCiWHkKZ0ljFw0yzu4Hg'
    output_path = os.path.join(files_dir, 'best_han_model.pth')
    gdown.download(url, output_path, quiet=False)

download_file()
  # Check for confirmation token in cookies (for large files)



# # import os
# # import shutil
# # import kagglehub

# # # Download latest version
# # path = kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset")

# # # Create 'data' directory if it doesn't exist
# # os.makedirs('data', exist_ok=True)

# # # Move all CSV files from the dataset path to the 'data' directory
# # for filename in os.listdir(path):
# #     if filename.endswith('.csv'):
# #         full_file_path = os.path.join(path, filename)
# #         shutil.copy(full_file_path, 'data')

# # print("CSV files saved to 'data/' directory.")
