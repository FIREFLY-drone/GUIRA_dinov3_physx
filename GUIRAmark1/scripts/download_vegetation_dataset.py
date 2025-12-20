import os
import requests
import zipfile
from tqdm import tqdm

# Placeholder URL
DATASET_URL = "https://example.com/vegetation_dataset.zip"
DOWNLOAD_PATH = "/workspaces/FIREPREVENTION/data/raw"
EXTRACT_PATH = "/workspaces/FIREPREVENTION/data/raw/vegetation_health"

def download_file(url, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    with open(path, 'wb') as file, tqdm(total=total_size, unit='iB', unit_scale=True) as bar:
        for data in response.iter_content(1024):
            file.write(data)
            bar.update(len(data))

def extract_zipfile(path, extract_path):
    if not os.path.exists(extract_path):
        os.makedirs(extract_path, exist_ok=True)
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"Extracted {path} to {extract_path}")

if __name__ == "__main__":
    file_path = os.path.join(DOWNLOAD_PATH, "vegetation_dataset.zip")
    print(f"Downloading dataset from {DATASET_URL} to {file_path}")
    download_file(DATASET_URL, file_path)
    print(f"Extracting dataset...")
    extract_zipfile(file_path, EXTRACT_PATH)
    print("Dataset downloaded and extracted successfully.")
