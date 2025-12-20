import os
import requests
import tarfile
from tqdm import tqdm

# This is a placeholder URL. Replace with the actual dataset URL.
DATASET_URL = "https://example.com/fire_detection_dataset.tar.gz"
DOWNLOAD_PATH = "/workspaces/FIREPREVENTION/data/raw"
EXTRACT_PATH = "/workspaces/FIREPREVENTION/data/raw/fire_detection"

def download_file(url, path):
    """Downloads a file from a URL to a given path with a progress bar."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte

    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")

def extract_tarfile(path, extract_path):
    """Extracts a tar.gz file."""
    if not os.path.exists(extract_path):
        os.makedirs(extract_path, exist_ok=True)
    
    with tarfile.open(path, "r:gz") as tar:
        tar.extractall(path=extract_path)
    print(f"Extracted {path} to {extract_path}")

if __name__ == "__main__":
    file_path = os.path.join(DOWNLOAD_PATH, "fire_detection_dataset.tar.gz")
    
    print(f"Downloading dataset from {DATASET_URL} to {file_path}")
    download_file(DATASET_URL, file_path)
    
    print(f"Extracting dataset...")
    extract_tarfile(file_path, EXTRACT_PATH)
    
    print("Dataset downloaded and extracted successfully.")
