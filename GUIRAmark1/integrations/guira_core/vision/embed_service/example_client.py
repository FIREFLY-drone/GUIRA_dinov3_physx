#!/usr/bin/env python3
"""
Example client for testing the DINOv3 Embedding Service.

Usage:
    python example_client.py --image path/to/image.jpg
"""

import argparse
import requests
import json
from pathlib import Path


def extract_embeddings(image_path: str, api_url: str = "http://localhost:8000"):
    """Extract embeddings from an image using the DINOv3 service.
    
    Args:
        image_path: Path to the image file
        api_url: Base URL of the embedding service
        
    Returns:
        Response dict with embedding_uri and metadata
    """
    print(f"📸 Processing image: {image_path}")
    
    # Check health first
    health_response = requests.get(f"{api_url}/health")
    if health_response.status_code == 200:
        health = health_response.json()
        print(f"✓ Service healthy: {health['model']} on {health['device']}")
    else:
        print(f"❌ Service not healthy")
        return None
    
    # Upload image for embedding extraction
    with open(image_path, 'rb') as f:
        files = {'file': (Path(image_path).name, f, 'image/jpeg')}
        response = requests.post(f"{api_url}/embed", files=files)
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ Embedding extraction successful!")
        print(f"  URI: {result['embedding_uri']}")
        print(f"  Shape: {result['shape']}")
        print(f"  Number of tiles: {result['num_tiles']}")
        print(f"\nMetadata:")
        for key, value in result['metadata'].items():
            print(f"  {key}: {value}")
        return result
    else:
        print(f"❌ Embedding extraction failed: {response.status_code}")
        print(f"  Error: {response.text}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Test DINOv3 embedding service")
    parser.add_argument(
        "--image",
        type=str,
        default="../../samples/sample_data/sample.jpg",
        help="Path to input image"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the embedding service"
    )
    
    args = parser.parse_args()
    
    if not Path(args.image).exists():
        print(f"❌ Image not found: {args.image}")
        return 1
    
    result = extract_embeddings(args.image, args.url)
    
    if result:
        print("\n🎉 Success! Embeddings saved to storage.")
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit(main())
