import os
import glob
import cv2
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shutil

# --- Configuration ---
PROCESSED_DIR = "/workspaces/FIREPREVENTION/data/processed/fire_yolo"
RAW_DATA_ROOT = "/workspaces/FIREPREVENTION/data/raw"
MANIFEST_DIR = "/workspaces/FIREPREVENTION/data/manifests"
CLASS_MAP = {"fire": 0, "smoke": 1}
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}

# Tiling configuration
TILE_SIZE = 640
STRIDE = 0.5 # 50% overlap
MIN_BOX_COVERAGE = 0.6

# --- Helper Functions ---

def clear_processed_dir():
    """Clears the processed data directory to ensure a fresh start."""
    if os.path.exists(PROCESSED_DIR):
        shutil.rmtree(PROCESSED_DIR)
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(PROCESSED_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(PROCESSED_DIR, 'labels', split), exist_ok=True)

def get_flame_rgb_data():
    """Extracts data from the FLAME dataset."""
    print("Processing FLAME dataset...")
    base_path = os.path.join(RAW_DATA_ROOT, 'flame_rgb/Output/Classification')
    image_files = []
    # The dataset has 'Fire' and 'No_Fire' folders
    for class_name in ['Fire', 'No_Fire']:
        class_path = os.path.join(base_path, class_name)
        if os.path.isdir(class_path):
            image_files.extend(glob.glob(os.path.join(class_path, '*.jpg')))
    
    # For FLAME, we only have image-level labels. We'll treat 'Fire' images as having fire.
    # This is a simplification; a proper approach would use the segmentation masks.
    all_data = []
    for img_path in image_files:
        label = [CLASS_MAP['fire']] if 'Fire' in img_path else []
        # Since we don't have bounding boxes, we can't create YOLO labels yet.
        # This dataset is better for classification or if we generate boxes from masks.
        # For now, we will skip creating YOLO data from this, but it can be used for hard negatives.
    print(f"FLAME dataset provides {len(image_files)} images, mainly for classification or hard negatives.")
    return [] # Returning empty as we need bbox labels for YOLO

def tile_image(image_path, boxes, image_id):
    """Tiles a large image and adjusts bounding boxes."""
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    tiles = []

    for y in range(0, h, int(TILE_SIZE * (1 - STRIDE))):
        for x in range(0, w, int(TILE_SIZE * (1 - STRIDE))):
            y_end, x_end = min(y + TILE_SIZE, h), min(x + TILE_SIZE, w)
            tile_img = img[y:y_end, x:x_end]
            
            tile_boxes = []
            for box in boxes:
                class_id, cx, cy, bw, bh = box
                # Convert YOLO to pixel coords
                abs_cx, abs_cy = cx * w, cy * h
                abs_bw, abs_bh = bw * w, bh * h
                x1, y1 = abs_cx - abs_bw / 2, abs_cy - abs_bh / 2
                x2, y2 = x1 + abs_bw, y1 + abs_bh

                # Check for intersection
                inter_x1, inter_y1 = max(x1, x), max(y1, y)
                inter_x2, inter_y2 = min(x2, x_end), min(y2, y_end)

                if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                    # Check for coverage
                    box_area = abs_bw * abs_bh
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    if inter_area / box_area > MIN_BOX_COVERAGE:
                        # New box coords relative to tile
                        new_x1, new_y1 = inter_x1 - x, inter_y1 - y
                        new_x2, new_y2 = inter_x2 - x, inter_y2 - y
                        
                        # Convert back to YOLO format
                        tile_h, tile_w, _ = tile_img.shape
                        new_cx = (new_x1 + new_x2) / (2 * tile_w)
                        new_cy = (new_y1 + new_y2) / (2 * tile_h)
                        new_bw = (new_x2 - new_x1) / tile_w
                        new_bh = (new_y2 - new_y1) / tile_h
                        tile_boxes.append([class_id, new_cx, new_cy, new_bw, new_bh])
            
            if tile_boxes:
                tile_name = f"{image_id}_tile_{y}_{x}"
                tiles.append({"image": tile_img, "boxes": tile_boxes, "name": tile_name})
    return tiles

def save_split(data, split):
    """Saves a data split to the processed directory."""
    for item in tqdm(data, desc=f"Saving {split} split"):
        img_path = os.path.join(PROCESSED_DIR, 'images', split, f"{item['name']}.jpg")
        lbl_path = os.path.join(PROCESSED_DIR, 'labels', split, f"{item['name']}.txt")
        
        cv2.imwrite(img_path, item['image'])
        with open(lbl_path, 'w') as f:
            for box in item['boxes']:
                f.write(f"{int(box[0])} {' '.join(map(str, box[1:]))}\n")

def create_final_manifest():
    """Creates the data.yaml file for YOLOv8."""
    data_yaml = {
        'path': os.path.abspath(PROCESSED_DIR),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(CLASS_MAP),
        'names': {k: v for k, v in INV_CLASS_MAP.items()}
    }
    with open(os.path.join(MANIFEST_DIR, 'fire_yolo.yaml'), 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)
    print("Created final data.yaml manifest.")

def convert_all_fire_datasets():
    """Main function to convert all fire-related datasets."""
    clear_processed_dir()
    
    all_tiled_data = []
    
    # In a real scenario, you would implement functions like `get_flame_rgb_data`
    # for each dataset to extract image paths and their corresponding bounding boxes.
    # This is a placeholder for where that logic would go.
    # For now, we'll create some dummy data to demonstrate the tiling and saving process.
    
    print("Simulating data processing (no real datasets are fully processed in this example).")
    dummy_image = np.random.randint(0, 255, size=(1024, 1024, 3), dtype=np.uint8)
    dummy_boxes = [
        [CLASS_MAP['fire'], 0.5, 0.5, 0.2, 0.2],
        [CLASS_MAP['smoke'], 0.2, 0.2, 0.1, 0.1]
    ]
    cv2.imwrite("dummy_large_image.jpg", dummy_image)
    
    tiled_data = tile_image("dummy_large_image.jpg", dummy_boxes, "dummy_image_1")
    all_tiled_data.extend(tiled_data)
    os.remove("dummy_large_image.jpg")

    if not all_tiled_data:
        print("Warning: No data was processed. Please check the dataset paths and processing functions.")
        return

    # Split data
    train_val, test = train_test_split(all_tiled_data, test_size=0.1, random_state=42)
    train, val = train_test_split(train_val, test_size=0.2, random_state=42)

    # Save splits
    save_split(train, 'train')
    save_split(val, 'val')
    save_split(test, 'test')

    # Create final manifest
    create_final_manifest()

if __name__ == '__main__':
    convert_all_fire_datasets()
