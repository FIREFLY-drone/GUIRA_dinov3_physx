import os
import requests
import zipfile
from tqdm import tqdm

# This is a placeholder URL. Replace with the actual dataset URL.
DATASET_URL = "https://example.com/fauna_dataset.zip"
DOWNLOAD_PATH = "/workspaces/FIREPREVENTION/data/raw"
EXTRACT_PATH = "/workspaces/FIREPREVENTION/data/raw/fauna_detection"

def download_file(url, path):
    """Downloads a file from a URL to a given path with a progress bar."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024

    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

def extract_zipfile(path, extract_path):
    """Extracts a zip file."""
    if not os.path.exists(extract_path):
        os.makedirs(extract_path, exist_ok=True)
    
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"Extracted {path} to {extract_path}")

if __name__ == "__main__":
    file_path = os.path.join(DOWNLOAD_PATH, "fauna_dataset.zip")
    
    print(f"Downloading dataset from {DATASET_URL} to {file_path}")
    download_file(DATASET_URL, file_path)
    
    print(f"Extracting dataset...")
    extract_zipfile(file_path, EXTRACT_PATH)
    
    print("Dataset downloaded and extracted successfully.")
