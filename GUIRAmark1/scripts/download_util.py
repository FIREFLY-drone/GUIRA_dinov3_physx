import os
import requests
import zipfile
import tarfile
import hashlib
import json
from tqdm import tqdm

CHECKSUM_FILE = "/workspaces/FIREPREVENTION/checksums.json"
DATA_DIR = "/workspaces/FIREPREVENTION/data/raw"
DOCS_DIR = "/workspaces/FIREPREVENTION/docs/datasets"

def verify_checksum(file_path, expected_checksum):
    """Verifies the SHA256 checksum of a file."""
    if expected_checksum == "placeholder_checksum":
        print(f"Warning: Placeholder checksum for {os.path.basename(file_path)}. Skipping verification.")
        return True
        
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    
    return sha256_hash.hexdigest() == expected_checksum

def download_and_extract(dataset_id, url, file_type='zip'):
    """
    Downloads, verifies, and extracts a dataset. Idempotent.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    
    filename = url.split('/')[-1]
    file_path = os.path.join(DATA_DIR, filename)
    extract_path = os.path.join(DATA_DIR, dataset_id)

    if os.path.exists(extract_path):
        print(f"Dataset {dataset_id} already exists at {extract_path}. Skipping.")
        return

    # Download
    if not os.path.exists(file_path):
        print(f"Downloading {dataset_id} from {url}...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        with open(file_path, 'wb') as f, tqdm(total=total_size, unit='iB', unit_scale=True) as bar:
            for data in response.iter_content(1024):
                f.write(data)
                bar.update(len(data))
    else:
        print(f"File {filename} already downloaded.")

    # Verify
    with open(CHECKSUM_FILE, 'r') as f:
        checksums = json.load(f)
    if not verify_checksum(file_path, checksums.get(filename, "placeholder_checksum")):
        print(f"Checksum mismatch for {filename}. Please check the file.")
        # os.remove(file_path) # Optional: remove corrupted file
        return

    # Extract
    print(f"Extracting {filename}...")
    if file_type == 'zip':
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    elif file_type == 'tar.gz':
        with tarfile.open(file_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_path)
    
    print(f"Extraction complete for {dataset_id}.")
    
    # Document License
    document_license(extract_path, dataset_id)

def document_license(extract_path, dataset_id):
    """Finds license file and copies its content to the docs."""
    license_content = "License information not found."
    for root, _, files in os.walk(extract_path):
        for file in files:
            if file.lower().startswith(('license', 'readme')):
                try:
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        license_content = f.read()
                    break
                except Exception as e:
                    license_content = f"Could not read license file: {e}"
        if license_content != "License information not found.":
            break
            
    doc_path = os.path.join(DOCS_DIR, f"{dataset_id}_license.md")
    with open(doc_path, 'w') as f:
        f.write(f"# License for {dataset_id}\n\n")
        f.write("```\n")
        f.write(license_content)
        f.write("\n```")
    print(f"Documented license for {dataset_id}.")

def download_with_git(dataset_id, git_url):
    """Clones a git repository. Idempotent."""
    extract_path = os.path.join(DATA_DIR, dataset_id)
    if os.path.exists(extract_path):
        print(f"Dataset {dataset_id} already exists at {extract_path}. Skipping.")
        return
    
    print(f"Cloning {dataset_id} from {git_url}...")
    os.system(f"git clone {git_url} {extract_path}")
    
    document_license(extract_path, dataset_id)

if __name__ == '__main__':
    # This is a utility script, not meant to be run directly.
    # Example usage:
    # download_with_git('flame_rgb', 'https://github.com/AlirezaShamsoshoara/Fire-Detection-UAV-Aerial-Image-Classification-Segmentation-UnmannedAerialVehicle.git')
    pass
