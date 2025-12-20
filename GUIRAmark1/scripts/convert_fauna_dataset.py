import os
import yaml
import glob
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import train_test_split

# --- Configuration ---
RAW_DATA_DIR = "/workspaces/FIREPREVENTION/data/raw/fauna"
PROCESSED_DATA_DIR_YOLO = "/workspaces/FIREPREVENTION/data/processed/fauna_yolo"
PROCESSED_DATA_DIR_CSRNET = "/workspaces/FIREPREVENTION/data/processed/fauna_csrnet"
MANIFEST_DIR = "/workspaces/FIREPREVENTION/data/manifests"
LABEL_MAP_PATH = "/workspaces/FIREPREVENTION/config/fauna_labelmap.yaml"

# Create directories
os.makedirs(os.path.join(PROCESSED_DATA_DIR_YOLO, 'images/train'), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DATA_DIR_YOLO, 'images/val'), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DATA_DIR_YOLO, 'labels/train'), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DATA_DIR_YOLO, 'labels/val'), exist_ok=True)

os.makedirs(os.path.join(PROCESSED_DATA_DIR_CSRNET, 'images/train'), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DATA_DIR_CSRNET, 'images/val'), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DATA_DIR_CSRNET, 'density_maps/train'), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DATA_DIR_CSRNET, 'density_maps/val'), exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(MANIFEST_DIR, exist_ok=True)


# --- Label Unification ---
with open(LABEL_MAP_PATH, 'r') as f:
    label_map_config = yaml.safe_load(f)

# Create a flat mapping from any dataset's label to a canonical name and ID
canonical_labels = {}
raw_to_canonical_map = {}
i = 0
for canonical_name, sources in label_map_config.items():
    if canonical_name not in canonical_labels:
        canonical_labels[canonical_name] = i
        i += 1
    for source, source_labels in sources.items():
        for label in source_labels:
            raw_to_canonical_map[label] = canonical_name


def get_canonical_label_id(raw_label_str):
    canonical_name = raw_to_canonical_map.get(raw_label_str, 'other_animal')
    return canonical_labels[canonical_name]

# --- Main Conversion Logic ---
def convert_fauna_dataset():
    """
    Converts fauna datasets for YOLOv8 (detection) and CSRNet (density).
    """
    image_files = glob.glob(os.path.join(RAW_DATA_DIR, '**/*.jpg'), recursive=True)
    
    # Dummy hard negatives
    hard_negatives = [f for f in image_files if "background" in f]
    image_files = [f for f in image_files if "background" not in f]

    if not image_files:
        print("No image files found. Skipping conversion.")
        return

    train_val_files, test_files = train_test_split(image_files, test_size=0.1, random_state=42)
    train_files, val_files = train_test_split(train_val_files, test_size=0.2, random_state=42)

    process_split(train_files, 'train')
    process_split(val_files, 'val')
    # process_split(test_files, 'test') # test split is handled similarly

    create_yolo_yaml()

def process_split(files, split_name):
    for img_path in files:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        # Assuming label file has the same name but with .txt extension and is in a parallel 'labels' folder
        label_path = img_path.replace('.jpg', '.txt').replace('images', 'labels')

        # --- YOLO Processing ---
        # Copy image, resizing to 960 for consistency
        yolo_img_dir = os.path.join(PROCESSED_DATA_DIR_YOLO, f'images/{split_name}')
        img = cv2.imread(img_path)
        img_resized = cv2.resize(img, (960, 960))
        cv2.imwrite(os.path.join(yolo_img_dir, f"{base_name}.jpg"), img_resized)

        # Process and write YOLO label
        yolo_label_dir = os.path.join(PROCESSED_DATA_DIR_YOLO, f'labels/{split_name}')
        if os.path.exists(label_path):
            with open(label_path, 'r') as f_in, open(os.path.join(yolo_label_dir, f"{base_name}.txt"), 'w') as f_out:
                for line in f_in:
                    parts = line.strip().split()
                    # This part is tricky without knowing the exact raw label format.
                    # Assuming the first element is a string label, e.g., "deer"
                    raw_label_str = parts[0] 
                    canonical_id = get_canonical_label_id(raw_label_str)
                    f_out.write(f"{canonical_id} {' '.join(parts[1:])}\n")

        # --- CSRNet Processing ---
        # Copy image
        csrnet_img_dir = os.path.join(PROCESSED_DATA_DIR_CSRNET, f'images/{split_name}')
        cv2.imwrite(os.path.join(csrnet_img_dir, f"{base_name}.jpg"), img)

        # Create and save density map
        density_map_dir = os.path.join(PROCESSED_DATA_DIR_CSRNET, f'density_maps/{split_name}')
        if os.path.exists(label_path):
            density_map = create_density_map(img_path, label_path)
            np.save(os.path.join(density_map_dir, f"{base_name}.npy"), density_map)

def create_density_map(img_path, label_path, sigma=4.0):
    """
    Generates a density map from bounding box centers.
    Sigma can be adaptive based on object size.
    """
    image = cv2.imread(img_path)
    h, w, _ = image.shape
    density_map = np.zeros((h, w), dtype=np.float32)

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            x_center, y_center, box_w, box_h = map(float, parts[1:])
            x = int(x_center * w)
            y = int(y_center * h)
            
            # Adaptive sigma based on box size
            box_area = (box_w * w) * (box_h * h)
            adaptive_sigma = max(2.0, np.sqrt(box_area) / 10.0)

            if 0 <= x < w and 0 <= y < h:
                density_map[y, x] = 1.0
    
    # Apply Gaussian filter with adaptive sigma
    density_map = gaussian_filter(density_map, sigma=adaptive_sigma)
    return density_map

def create_yolo_yaml():
    data_yaml = {
        'train': os.path.abspath(os.path.join(PROCESSED_DATA_DIR_YOLO, 'images/train')),
        'val': os.path.abspath(os.path.join(PROCESSED_DATA_DIR_YOLO, 'images/val')),
        'nc': len(canonical_labels),
        'names': list(canonical_labels.keys())
    }
    with open(os.path.join(PROCESSED_DATA_DIR_YOLO, 'data.yaml'), 'w') as f:
        yaml.dump(data_yaml, f)

if __name__ == "__main__":
    # Create dummy raw data for testing
    os.makedirs(os.path.join(RAW_DATA_DIR, 'waid_fauna/images'), exist_ok=True)
    os.makedirs(os.path.join(RAW_DATA_DIR, 'waid_fauna/labels'), exist_ok=True)
    cv2.imwrite(os.path.join(RAW_DATA_DIR, 'waid_fauna/images/test1.jpg'), np.zeros((1024, 1024, 3), dtype=np.uint8))
    with open(os.path.join(RAW_DATA_DIR, 'waid_fauna/labels/test1.txt'), 'w') as f:
        # Assuming format: label_str x_center y_center w h
        f.write("elephant 0.5 0.5 0.1 0.1\n")

    convert_fauna_dataset()
    print("Fauna dataset conversion complete.")
