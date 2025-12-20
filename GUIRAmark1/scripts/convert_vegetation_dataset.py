import os
import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import yaml
import json

# --- Configuration ---
RAW_DATA_DIR = "/workspaces/FIREPREVENTION/data/raw/vegetation"
PROCESSED_DATA_DIR = "/workspaces/FIREPREVENTION/data/processed/vegetation_resnet_vari"
MANIFEST_DIR = "/workspaces/FIREPREVENTION/data/manifests"
# In a real scenario, this would point to a valid model file.
DEEPFOREST_MODEL_PATH = "path/to/your/deepforest_model.pt" 
# Health classes
CLASSES = {'healthy': 0, 'stressed': 1, 'burned': 2}

# Threshold for green channel to determine healthy vegetation
GREEN_THRESHOLD = 128
CLASSES = {'healthy': 0, 'stressed': 1, 'burned': 2}

# Create directories
os.makedirs(os.path.join(PROCESSED_DATA_DIR, 'train/healthy'), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DATA_DIR, 'train/stressed'), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DATA_DIR, 'train/burned'), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DATA_DIR, 'val/healthy'), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DATA_DIR, 'val/stressed'), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DATA_DIR, 'val/burned'), exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(MANIFEST_DIR, exist_ok=True)


def compute_vari(image):
    """
    Computes the Visible Atmospherically Resistant Index (VARI).
    VARI = (Green - Red) / (Green + Red - Blue)
    """
    # Ensure image is float for calculations
    B = image[:, :, 0].astype(np.float32)
    G = image[:, :, 1].astype(np.float32)
    R = image[:, :, 2].astype(np.float32)
    
    # To avoid division by zero
    denominator = G + R - B
    denominator[denominator == 0] = 1e-6
    
    vari = (G - R) / denominator
    
    # Normalize to 0-255 range to be saved as a grayscale image channel
    vari_normalized = cv2.normalize(vari, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return vari_normalized

def detect_crowns_placeholder(image_path):
    """
    Placeholder for DeepForest crown detection.
    In a real implementation, this would use the DeepForest library to predict bounding boxes.
    Returns a list of bounding boxes [x1, y1, x2, y2].
    """
    # This is a dummy implementation.
    img = cv2.imread(image_path)
    if img is None:
        return []
    h, w, _ = img.shape
    # Return a few random boxes to simulate crown detection
    return [
        (50, 50, 150, 150),
        (w - 200, h - 200, w - 100, h - 100),
        (w//2 - 60, h//2 - 60, w//2 + 60, h//2 + 60)
    ]

def convert_vegetation_dataset(vari_config='channel'):
    """
    Converts vegetation dataset by detecting crowns, computing VARI, and creating patches.
    'vari_config' can be 'channel' (stack VARI as 4th channel) or 'feature' (save separately).
    """
    image_files = glob.glob(os.path.join(RAW_DATA_DIR, '*.jpg'))
    if not image_files:
        print("No images found in raw vegetation data directory. Creating dummy data.")
        # Create dummy raw data if none exists
        dummy_healthy_img = np.full((500, 500, 3), (0, 200, 0), dtype=np.uint8)
        cv2.imwrite(os.path.join(RAW_DATA_DIR, 'healthy_scene.jpg'), dummy_healthy_img)

        dummy_stressed_img = np.full((500, 500, 3), (50, 100, 50), dtype=np.uint8)
        cv2.imwrite(os.path.join(RAW_DATA_DIR, 'stressed_scene.jpg'), dummy_stressed_img)
        
        dummy_burned_img = np.full((500, 500, 3), (30, 40, 100), dtype=np.uint8)
        cv2.imwrite(os.path.join(RAW_DATA_DIR, 'burned_scene.jpg'), dummy_burned_img)
        image_files = glob.glob(os.path.join(RAW_DATA_DIR, '*.jpg'))


    train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)

    process_split(train_files, 'train', vari_config)
    process_split(val_files, 'val', vari_config)

    print("Vegetation dataset conversion complete.")

def process_split(files, split_name, vari_config):
    manifest = []
    for img_path in files:
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        crown_boxes = detect_crowns_placeholder(img_path)
        vari_channel = compute_vari(image)

        for i, box in enumerate(crown_boxes):
            x1, y1, x2, y2 = box
            crown_patch = image[y1:y2, x1:x2]
            
            # Heuristic for labeling based on filename or patch color
            if 'burned' in os.path.basename(img_path):
            elif np.mean(crown_patch[:, :, 1]) > GREEN_THRESHOLD: # High green value
                label = 'healthy'
                label = 'healthy'
            else:
                label = 'stressed'

            base_name = os.path.splitext(os.path.basename(img_path))[0]
            patch_filename = f"{base_name}_crown_{i}.png"
            save_dir = os.path.join(PROCESSED_DATA_DIR, split_name, label)
            save_path = os.path.join(save_dir, patch_filename)

            if vari_config == 'channel':
                vari_patch = vari_channel[y1:y2, x1:x2]
                vari_patch_resized = cv2.resize(vari_patch, (crown_patch.shape[1], crown_patch.shape[0]))
                # Stack as BGR + VARI
                output_patch = np.dstack((crown_patch, vari_patch_resized))
                cv2.imwrite(save_path, output_patch)
                manifest.append({'path': save_path, 'label': CLASSES[label]})

            elif vari_config == 'feature':
                # Save RGB image
                cv2.imwrite(save_path, crown_patch)
                # Save VARI feature separately
                vari_patch = vari_channel[y1:y2, x1:x2]
                vari_save_path = save_path.replace('.png', '_vari.npy')
                np.save(vari_save_path, vari_patch)
                manifest.append({'path': save_path, 'vari_path': vari_save_path, 'label': CLASSES[label]})

    # Write manifest for the split
    with open(os.path.join(MANIFEST_DIR, f'vegetation_{split_name}_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=4)


if __name__ == "__main__":
    # Example of running the conversion with VARI as an extra channel
    convert_vegetation_dataset(vari_config='channel')
