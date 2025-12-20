# Create sample DEM data
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import os

# Create sample DEM data
def create_sample_dem():
    # DEM parameters
    width, height = 1000, 1000
    xmin, ymin, xmax, ymax = -122.5, 37.7, -122.3, 37.9  # San Francisco Bay Area coordinates
    
    # Create elevation data (meters)
    # Simulate mountainous terrain
    x = np.linspace(0, 10, width)
    y = np.linspace(0, 10, height)
    X, Y = np.meshgrid(x, y)
    
    # Create realistic elevation pattern
    elevation = (
        200 * np.sin(X * 0.8) * np.cos(Y * 0.6) +
        150 * np.sin(X * 1.2 + Y * 0.8) +
        100 * np.cos(X * 0.5 + Y * 1.1) +
        300  # Base elevation
    )
    
    # Add noise for realism
    elevation += np.random.normal(0, 10, elevation.shape)
    
    # Ensure positive elevation
    elevation = np.maximum(elevation, 0)
    
    # Create transform
    transform = from_bounds(xmin, ymin, xmax, ymax, width, height)
    
    # Create DEM file
    dem_path = "data/dem/sample_dem.tif"
    os.makedirs("data/dem", exist_ok=True)
    
    with rasterio.open(
        dem_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=elevation.dtype,
        crs='EPSG:4326',
        transform=transform,
    ) as dst:
        dst.write(elevation, 1)
    
    print(f"Created sample DEM: {dem_path}")
    print(f"Elevation range: {elevation.min():.1f} - {elevation.max():.1f} meters")
    
    return dem_path

# Create sample DEM
dem_file = create_sample_dem()

# Create sample intrinsics file
intrinsics_data = {
    "fx": 800.0,
    "fy": 800.0,
    "cx": 320.0,
    "cy": 240.0,
    "k1": -0.1,
    "k2": 0.05,
    "p1": 0.001,
    "p2": -0.001,
    "image_width": 640,
    "image_height": 480
}

import json
os.makedirs("config", exist_ok=True)
with open("config/intrinsics.json", 'w') as f:
    json.dump(intrinsics_data, f, indent=2)

print("Created sample camera intrinsics file")

# Create sample video sequence data for smoke detection
def create_sample_video_data():
    os.makedirs("data/smoke/sequences", exist_ok=True)
    
    # Create sample video sequence (simulate frame names)
    sequence_dir = "data/smoke/sequences/seq_001"
    os.makedirs(sequence_dir, exist_ok=True)
    
    # Create 16 sample frames (TimeSFormer typically uses 8-16 frames)
    for i in range(16):
        frame_path = f"{sequence_dir}/frame_{i:04d}.jpg"
        
        # Create simple frame with potential smoke
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Add some smoke-like patterns (gray, moving)
        if i > 5:  # Smoke appears after frame 5
            smoke_intensity = min(50 + i * 10, 150)
            smoke_region = np.random.randint(0, 224, (100, 2))
            for x, y in smoke_region:
                if 0 <= x < 224 and 0 <= y < 224:
                    img[x, y] = [smoke_intensity, smoke_intensity, smoke_intensity]
        
        # Add background
        background_mask = np.all(img == [0, 0, 0], axis=2)
        img[background_mask] = [50, 100, 50]  # Dark green background
        
        from PIL import Image
        Image.fromarray(img).save(frame_path)
    
    # Create label file for sequence
    with open(f"{sequence_dir}/label.txt", 'w') as f:
        f.write("1\n")  # 1 = smoke present, 0 = no smoke
    
    print(f"Created sample video sequence: {sequence_dir}")

create_sample_video_data()

# Create sample fauna data
def create_sample_fauna_data():
    os.makedirs("data/fauna/images", exist_ok=True)
    os.makedirs("data/fauna/labels", exist_ok=True)
    os.makedirs("data/fauna/health", exist_ok=True)
    
    from PIL import Image
    
    for i in range(5):
        # Create sample image with animals
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Add background (forest)
        img[:, :] = [34, 139, 34]  # Forest green
        
        # Add animal-like region (brown/tan)
        animal_x = np.random.randint(100, 500)
        animal_y = np.random.randint(100, 500)
        animal_size = np.random.randint(30, 80)
        
        y, x = np.ogrid[:640, :640]
        animal_mask = ((x - animal_x)**2 + (y - animal_y)**2) < animal_size**2
        img[animal_mask] = [139, 69, 19]  # Brown for animal
        
        # Save image
        img_path = f"data/fauna/images/fauna_{i+1:03d}.jpg"
        Image.fromarray(img).save(img_path)
        
        # Create YOLO label (animal detection)
        label_path = f"data/fauna/labels/fauna_{i+1:03d}.txt"
        with open(label_path, 'w') as f:
            # Normalize coordinates
            center_x = animal_x / 640
            center_y = animal_y / 640
            width = height = (animal_size * 2) / 640
            f.write(f"0 {center_x:.3f} {center_y:.3f} {width:.3f} {height:.3f}\n")
        
        # Create health assessment (0=healthy, 1=stressed, 2=injured)
        health_path = f"data/fauna/health/fauna_{i+1:03d}.txt"
        with open(health_path, 'w') as f:
            health_status = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1])
            f.write(f"{health_status}\n")
    
    print("Created sample fauna detection and health data")

create_sample_fauna_data()

# Create sample vegetation data
def create_sample_vegetation_data():
    os.makedirs("data/vegetation/images", exist_ok=True)
    os.makedirs("data/vegetation/health", exist_ok=True)
    
    from PIL import Image
    
    for i in range(5):
        # Create vegetation image with varying health
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        
        health_level = np.random.choice([0, 1, 2, 3])  # 0=healthy, 1=stressed, 2=diseased, 3=dead
        
        if health_level == 0:  # Healthy
            img[:, :] = [34, 139, 34]  # Bright green
        elif health_level == 1:  # Stressed
            img[:, :] = [107, 142, 35]  # Olive green
        elif health_level == 2:  # Diseased
            img[:, :] = [160, 82, 45]  # Brown
        else:  # Dead
            img[:, :] = [101, 67, 33]  # Dark brown
        
        # Add some variation
        noise = np.random.randint(-30, 30, img.shape)
        img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
        
        # Save image
        img_path = f"data/vegetation/images/veg_{i+1:03d}.jpg"
        Image.fromarray(img).save(img_path)
        
        # Save health label
        health_path = f"data/vegetation/health/veg_{i+1:03d}.txt"
        with open(health_path, 'w') as f:
            f.write(f"{health_level}\n")
        
        # Calculate and save VARI index
        r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # VARI = (Green - Red) / (Green + Red - Blue)
        vari = (g.astype(float) - r.astype(float)) / (g.astype(float) + r.astype(float) - b.astype(float) + 1e-8)
        vari_mean = np.mean(vari)
        
        vari_path = f"data/vegetation/health/veg_{i+1:03d}_vari.txt"
        with open(vari_path, 'w') as f:
            f.write(f"{vari_mean:.6f}\n")
    
    print("Created sample vegetation health data with VARI indices")

create_sample_vegetation_data()

print("\n✅ Sample data creation completed!")
print("Created directories and files for:")
print("- Fire detection (images + YOLO labels)")
print("- Smoke detection (video sequences)")
print("- Fauna detection (images + health status)")
print("- Vegetation health (images + VARI indices)")
print("- DEM data (elevation model)")
print("- Camera intrinsics (calibration)")
print("- Drone pose data (existing)")
