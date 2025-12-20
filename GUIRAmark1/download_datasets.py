"""
Dataset Download Script for Fire Prevention System
Downloads and prepares all required datasets for training.
"""

import argparse
import os
import sys
import zipfile
import tarfile
from pathlib import Path
from urllib.request import urlretrieve
import json
import csv
from tqdm import tqdm
import requests
import gdown

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from utils import setup_logging, load_config
from loguru import logger


class DatasetDownloader:
    """Downloads and prepares datasets for the fire prevention system."""
    
    def __init__(self, config_path: str = 'config.yaml'):
        self.config = load_config(config_path)
        self.data_dir = Path('data')
        self.data_dir.mkdir(exist_ok=True)
        
        # Dataset URLs from config
        self.datasets = self.config.get('datasets', {})
        
    def download_with_progress(self, url: str, filepath: str):
        """Download file with progress bar."""
        try:
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f, tqdm(
                desc=f"Downloading {Path(filepath).name}",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
                        
            return True
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False
    
    def extract_archive(self, filepath: str, extract_to: str):
        """Extract zip or tar archive."""
        filepath = Path(filepath)
        extract_to = Path(extract_to)
        
        try:
            if filepath.suffix.lower() == '.zip':
                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            elif filepath.suffix.lower() in ['.tar', '.gz', '.tgz']:
                with tarfile.open(filepath, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_to)
            else:
                logger.warning(f"Unknown archive format: {filepath}")
                return False
                
            logger.info(f"Extracted {filepath} to {extract_to}")
            return True
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return False
    
    def download_fire_datasets(self):
        """Download fire detection datasets."""
        logger.info("Downloading fire detection datasets...")
        
        fire_dir = self.data_dir / 'fire'
        fire_dir.mkdir(exist_ok=True)
        
        # Sample fire dataset (placeholder - replace with actual URLs)
        fire_urls = [
            {
                'name': 'Fire_Sample_Images',
                'url': 'https://example.com/fire_images.zip',  # Replace with actual URL
                'description': 'Sample fire detection images'
            }
        ]
        
        for dataset in fire_urls:
            logger.info(f"Downloading {dataset['name']}...")
            
            # Create download path
            download_path = fire_dir / f"{dataset['name']}.zip"
            
            # Skip if already exists
            if download_path.exists():
                logger.info(f"Dataset {dataset['name']} already exists, skipping...")
                continue
            
            # Download (using placeholder for now)
            logger.warning(f"Dataset URL is placeholder - {dataset['description']}")
            logger.info("Please manually download fire detection datasets and place them in data/fire/")
            logger.info("Required structure: data/fire/images/ and data/fire/labels/")
            
        # Create sample structure
        (fire_dir / 'images').mkdir(exist_ok=True)
        (fire_dir / 'labels').mkdir(exist_ok=True)
        
        # Create sample annotation format
        sample_annotation = {
            "images": [
                {
                    "id": 1,
                    "file_name": "sample_fire_001.jpg",
                    "width": 640,
                    "height": 480
                }
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [100, 100, 200, 150],
                    "area": 30000,
                    "iscrowd": 0
                }
            ],
            "categories": [
                {"id": 1, "name": "fire"},
                {"id": 2, "name": "smoke"}
            ]
        }
        
        with open(fire_dir / 'annotations_sample.json', 'w') as f:
            json.dump(sample_annotation, f, indent=2)
            
        logger.info("Created sample fire dataset structure")
    
    def download_smoke_datasets(self):
        """Download smoke detection datasets."""
        logger.info("Downloading smoke detection datasets...")
        
        smoke_dir = self.data_dir / 'smoke'
        smoke_dir.mkdir(exist_ok=True)
        
        # Create directories
        (smoke_dir / 'videos').mkdir(exist_ok=True)
        
        # Create sample annotation CSV
        sample_annotations = [
            ['video_name', 'frame_index', 'smoke_flag'],
            ['sample_video_001.mp4', 0, 0],
            ['sample_video_001.mp4', 1, 0],
            ['sample_video_001.mp4', 2, 1],
            ['sample_video_002.mp4', 0, 1],
            ['sample_video_002.mp4', 1, 1]
        ]
        
        with open(smoke_dir / 'annotations.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(sample_annotations)
        
        logger.info("Created sample smoke dataset structure")
        logger.info("Please place video files in data/smoke/videos/")
    
    def download_fauna_datasets(self):
        """Download fauna detection datasets."""
        logger.info("Downloading fauna detection datasets...")
        
        fauna_dir = self.data_dir / 'fauna'
        fauna_dir.mkdir(exist_ok=True)
        
        # Create directories
        (fauna_dir / 'images').mkdir(exist_ok=True)
        
        # Create sample COCO format annotation
        sample_annotation = {
            "images": [
                {
                    "id": 1,
                    "file_name": "deer_001.jpg",
                    "width": 1024,
                    "height": 768
                }
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [200, 150, 300, 400],
                    "area": 120000,
                    "iscrowd": 0,
                    "health": "healthy",
                    "species": "deer"
                }
            ],
            "categories": [
                {"id": 1, "name": "deer"},
                {"id": 2, "name": "elk"},
                {"id": 3, "name": "bear"},
                {"id": 4, "name": "bird"},
                {"id": 5, "name": "other"}
            ]
        }
        
        with open(fauna_dir / 'annotations.json', 'w') as f:
            json.dump(sample_annotation, f, indent=2)
            
        logger.info("Created sample fauna dataset structure")
        logger.info("Please place wildlife images in data/fauna/images/")
    
    def download_vegetation_datasets(self):
        """Download vegetation health datasets."""
        logger.info("Downloading vegetation health datasets...")
        
        veg_dir = self.data_dir / 'vegetation'
        veg_dir.mkdir(exist_ok=True)
        
        # Create directories
        (veg_dir / 'images').mkdir(exist_ok=True)
        (veg_dir / 'masks').mkdir(exist_ok=True)
        
        logger.info("Created sample vegetation dataset structure")
        logger.info("Please place vegetation images in data/vegetation/images/")
        logger.info("Please place health masks in data/vegetation/masks/")
        logger.info("Mask format: 0=healthy, 1=dry, 2=burned")
    
    def create_pose_data(self):
        """Create sample pose data."""
        logger.info("Creating sample pose data...")
        
        pose_dir = self.data_dir / 'pose'
        pose_dir.mkdir(exist_ok=True)
        
        # Create sample pose CSV
        sample_pose_data = [
            ['frame', 'lat', 'lon', 'alt', 'yaw', 'pitch', 'roll'],
            [0, 40.7128, -74.0060, 100.0, 0.0, -10.0, 2.0],
            [1, 40.7129, -74.0061, 101.0, 1.0, -9.5, 1.8],
            [2, 40.7130, -74.0062, 102.0, 2.0, -9.0, 1.5],
            [3, 40.7131, -74.0063, 103.0, 3.0, -8.5, 1.2],
            [4, 40.7132, -74.0064, 104.0, 4.0, -8.0, 1.0]
        ]
        
        with open(pose_dir / 'pose.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(sample_pose_data)
            
        logger.info("Created sample pose data")
    
    def create_dem_data(self):
        """Create sample DEM data structure."""
        logger.info("Creating DEM data structure...")
        
        dem_dir = self.data_dir / 'dem'
        dem_dir.mkdir(exist_ok=True)
        
        # Create README for DEM data
        dem_readme = """
# Digital Elevation Model (DEM) Data

Place GeoTIFF (.tif) elevation files in this directory.

Required format:
- GeoTIFF format (.tif)
- Coordinate Reference System (CRS) information
- Elevation values in meters

Example files:
- elevation_tile_001.tif
- elevation_tile_002.tif

You can download DEM data from:
- USGS Earth Explorer (https://earthexplorer.usgs.gov/)
- NASA SRTM (https://www2.jpl.nasa.gov/srtm/)
- OpenTopography (https://www.opentopography.org/)
"""
        
        with open(dem_dir / 'README.md', 'w') as f:
            f.write(dem_readme)
            
        logger.info("Created DEM data structure")
    
    def create_spread_data(self):
        """Create sample fire spread data."""
        logger.info("Creating fire spread simulation data...")
        
        spread_dir = self.data_dir / 'spread' / 'historical'
        spread_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample fire spread data
        sample_data = {
            "fire_events": [
                {
                    "id": "fire_001",
                    "ignition_point": [40.7128, -74.0060],
                    "weather": {
                        "wind_speed": 15.0,
                        "wind_direction": 45.0,
                        "humidity": 0.3,
                        "temperature": 30.0
                    },
                    "spread_sequence": [
                        {"time": 0, "area": 100, "perimeter": [[40.7128, -74.0060]]},
                        {"time": 300, "area": 250, "perimeter": [[40.7129, -74.0061]]},
                        {"time": 600, "area": 400, "perimeter": [[40.7130, -74.0062]]}
                    ]
                }
            ]
        }
        
        with open(spread_dir / 'fire_events.json', 'w') as f:
            json.dump(sample_data, f, indent=2)
            
        logger.info("Created sample fire spread data")
    
    def download_all(self):
        """Download all datasets."""
        logger.info("Starting dataset download process...")
        
        try:
            self.download_fire_datasets()
            self.download_smoke_datasets()
            self.download_fauna_datasets()
            self.download_vegetation_datasets()
            self.create_pose_data()
            self.create_dem_data()
            self.create_spread_data()
            
            logger.info("Dataset download process completed!")
            logger.info("Note: Some datasets require manual download due to licensing restrictions")
            logger.info("Please check the data/ directory for instructions")
            
        except Exception as e:
            logger.error(f"Dataset download failed: {e}")
            return False
            
        return True
    
    def verify_datasets(self):
        """Verify that all required datasets are present."""
        logger.info("Verifying dataset structure...")
        
        required_dirs = [
            'data/fire/images',
            'data/fire/labels',
            'data/smoke/videos',
            'data/fauna/images',
            'data/vegetation/images',
            'data/vegetation/masks',
            'data/pose',
            'data/dem',
            'data/spread/historical'
        ]
        
        missing_dirs = []
        for dir_path in required_dirs:
            if not Path(dir_path).exists():
                missing_dirs.append(dir_path)
        
        if missing_dirs:
            logger.warning("Missing directories:")
            for dir_path in missing_dirs:
                logger.warning(f"  - {dir_path}")
            return False
        else:
            logger.info("All required directories present")
            return True


def main():
    parser = argparse.ArgumentParser(description='Download datasets for fire prevention system')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--verify-only', action='store_true', help='Only verify dataset structure')
    parser.add_argument('--dataset', choices=['fire', 'smoke', 'fauna', 'vegetation', 'all'], 
                       default='all', help='Which dataset to download')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging('INFO')
    
    logger.info("Fire Prevention System - Dataset Downloader")
    logger.info("=" * 50)
    
    # Initialize downloader
    downloader = DatasetDownloader(args.config)
    
    if args.verify_only:
        success = downloader.verify_datasets()
        sys.exit(0 if success else 1)
    
    # Download specific dataset or all
    if args.dataset == 'all':
        success = downloader.download_all()
    elif args.dataset == 'fire':
        downloader.download_fire_datasets()
        success = True
    elif args.dataset == 'smoke':
        downloader.download_smoke_datasets()
        success = True
    elif args.dataset == 'fauna':
        downloader.download_fauna_datasets()
        success = True
    elif args.dataset == 'vegetation':
        downloader.download_vegetation_datasets()
        success = True
    
    # Verify after download
    if success:
        downloader.verify_datasets()
    
    logger.info("Dataset download script completed")


if __name__ == "__main__":
    main()
