# Fire Detection Sample Image
import numpy as np
from PIL import Image
import os

# Create sample fire images
def create_sample_fire_image(filename, has_fire=True):
    # Create a 640x640 RGB image
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    
    if has_fire:
        # Add fire-like regions (orange/red/yellow)
        fire_region_x = np.random.randint(100, 500)
        fire_region_y = np.random.randint(100, 500)
        fire_size = np.random.randint(50, 150)
        
        # Create fire gradient
        y, x = np.ogrid[:640, :640]
        center_dist = np.sqrt((x - fire_region_x)**2 + (y - fire_region_y)**2)
        fire_mask = center_dist < fire_size
        
        # Fire colors: red to orange to yellow
        img[fire_mask] = [255, np.random.randint(100, 200), 0]  # Orange-red
        
        # Add some smoke (gray regions)
        smoke_mask = (center_dist < fire_size * 1.5) & (center_dist >= fire_size)
        img[smoke_mask] = [128, 128, 128]  # Gray smoke
    
    # Add background (green vegetation)
    background_mask = np.all(img == [0, 0, 0], axis=2)
    img[background_mask] = [34, 139, 34]  # Forest green
    
    # Add some noise
    noise = np.random.randint(-20, 20, img.shape)
    img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
    
    # Save image
    Image.fromarray(img).save(filename)
    return filename

# Create sample data directory structure
os.makedirs("data/fire/images", exist_ok=True)
os.makedirs("data/fire/labels", exist_ok=True)

# Create sample fire images
fire_images = []
for i in range(5):
    img_path = f"data/fire/images/fire_{i+1:03d}.jpg"
    fire_images.append(create_sample_fire_image(img_path, has_fire=True))
    
    # Create corresponding YOLO format label
    label_path = f"data/fire/labels/fire_{i+1:03d}.txt"
    with open(label_path, 'w') as f:
        # YOLO format: class_id center_x center_y width height (normalized)
        f.write("0 0.5 0.5 0.3 0.3\n")  # Fire class in center with 30% width/height

# Create non-fire images
for i in range(5):
    img_path = f"data/fire/images/no_fire_{i+1:03d}.jpg"
    fire_images.append(create_sample_fire_image(img_path, has_fire=False))
    
    # Empty label file for no-fire images
    label_path = f"data/fire/labels/no_fire_{i+1:03d}.txt"
    with open(label_path, 'w') as f:
        pass  # Empty file

print("Created sample fire detection images and labels")
