import sys
import os

# Add scripts directory to path to import download_util
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from download_util import download_with_git, download_and_extract

def download_all():
    """
    Downloads all datasets required for the project.
    """
    print("--- Starting Dataset Acquisition ---")

    # Fire Datasets
    download_with_git('flame_rgb', 'https://github.com/AlirezaShamsoshoara/Fire-Detection-UAV-Aerial-Image-Classification-Segmentation-UnmannedAerialVehicle.git')
    download_with_git('flame_rgb_simplified', 'https://github.com/sunnyiisc/Fire-Detection-from-FLAME-Dataset.git')
    download_with_git('flame2_rgb_ir', 'https://github.com/xiwenc1/Flame_2_dataset.git')
    download_with_git('sfgdn_fire', 'https://github.com/mi-luo/Flame-detection.git')
    
    # Fauna Datasets
    download_with_git('waid_fauna', 'https://github.com/xiaohuicui/WAID.git')
    
    # Placeholder for datasets requiring manual download (e.g., Kaggle)
    print("\n--- Manual Downloads Required ---")
    print("Please download the following datasets and place them in 'data/raw/':")
    print("- kaggle_fauna: https://www.kaggle.com/datasets/sugamg/wildlife-aerial-imagery-dataset")
    print("- iSAID: https://captain-whu.github.io/iSAID/dataset.html")
    print("---------------------------------")
    
    # The other datasets are from arXiv papers without direct download links in the prompt.
    # These would need to be sourced manually.
    # download_and_extract('flame3_thermal', 'URL_FOR_FLAME3')
    # download_and_extract('wit_uas_thermal', 'URL_FOR_WIT_UAS')
    # download_and_extract('awir_fauna', 'URL_FOR_AWIR')

if __name__ == '__main__':
    download_all()
