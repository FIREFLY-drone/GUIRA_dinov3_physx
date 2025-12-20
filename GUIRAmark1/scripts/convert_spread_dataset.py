import os
import numpy as np
import glob
from sklearn.model_selection import train_test_split
import json

# --- Configuration ---
RAW_DATA_DIR = "/workspaces/FIREPREVENTION/data/raw/spread"
PROCESSED_DATA_DIR = "/workspaces/FIREPREVENTION/data/processed/spread_hybrid"
MANIFEST_DIR = "/workspaces/FIREPREVENTION/data/manifests"

# Tiling and sequence parameters
TILE_SIZE = 256
T_IN = 6
T_OUT = 12

# Create directories
os.makedirs(os.path.join(PROCESSED_DATA_DIR, 'train'), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DATA_DIR, 'val'), exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(MANIFEST_DIR, exist_ok=True)


def convert_spread_dataset():
    """
    Processes raster data into sequences of tiles for the fire spread model.
    """
    # This assumes we have sequences of raster data.
    # Let's simulate this with numpy arrays.
    # A real implementation would use rasterio or gdal to read geo-referenced files.
    simulation_runs = glob.glob(os.path.join(RAW_DATA_DIR, 'sim_*'))
    if not simulation_runs:
        print("No simulation data found. Creating dummy data.")
        create_dummy_spread_data()
        simulation_runs = glob.glob(os.path.join(RAW_DATA_DIR, 'sim_*'))

    all_sequences = []
    for sim_run_dir in simulation_runs:
        # This would be a list of multi-band raster files sorted by time
        raster_files = sorted(glob.glob(os.path.join(sim_run_dir, '*.npy')))
        
        # Load all data for a simulation run into memory
        # In practice, this might need to be done more efficiently for large datasets
        full_raster_stack = np.array([np.load(f) for f in raster_files])
        
        # Tile the entire stack
        sequences = tile_and_sequence_data(full_raster_stack, sim_run_dir)
        all_sequences.extend(sequences)

    # Split sequences into train/val
    train_seq, val_seq = train_test_split(all_sequences, test_size=0.2, random_state=42)

    # Save the processed data and create manifests
    save_split(train_seq, 'train')
    save_split(val_seq, 'val')

    print("Fire spread dataset conversion complete.")

def tile_and_sequence_data(raster_stack, sim_run_dir):
    """
    Tiles the data and creates input/output sequences.
    """
    num_timesteps, num_channels, h, w = raster_stack.shape
    sequences = []
    
    stride = TILE_SIZE // 2 # Overlapping tiles

    for y in range(0, h - TILE_SIZE + 1, stride):
        for x in range(0, w - TILE_SIZE + 1, stride):
            for t in range(num_timesteps - (T_IN + T_OUT) + 1):
                
                input_seq_stack = raster_stack[t:t+T_IN, :, y:y+TILE_SIZE, x:x+TILE_SIZE]
                output_seq_stack = raster_stack[t+T_IN:t+T_IN+T_OUT, :, y:y+TILE_SIZE, x:x+TILE_SIZE]

                # We only care about the 'burn mask' for the output
                # Assuming burn mask is the last channel
                output_burn_mask = output_seq_stack[:, -1, :, :]

                sequence_info = {
                    'input_stack': input_seq_stack,
                    'output_mask': output_burn_mask,
                    'sim_run': os.path.basename(sim_run_dir),
                    'tile_coords': (y, x),
                    'start_time': t
                }
                sequences.append(sequence_info)
    return sequences

def save_split(sequences, split_name):
    """
    Saves the processed sequences to disk and creates a manifest.
    """
    manifest = []
    split_dir = os.path.join(PROCESSED_DATA_DIR, split_name)
    
    for i, seq_info in enumerate(sequences):
        input_filename = f"input_{i}.npy"
        output_filename = f"output_{i}.npy"
        
        input_path = os.path.join(split_dir, input_filename)
        output_path = os.path.join(split_dir, output_filename)
        
        np.save(input_path, seq_info['input_stack'])
        np.save(output_path, seq_info['output_mask'])
        
        manifest.append({
            'input_path': os.path.abspath(input_path),
            'output_path': os.path.abspath(output_path),
            'sim_run': seq_info['sim_run'],
            'tile_coords': seq_info['tile_coords'],
            'start_time': seq_info['start_time']
        })
        
    with open(os.path.join(MANIFEST_DIR, f'spread_{split_name}_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=4)

def create_dummy_spread_data():
    """Creates dummy raster data for testing."""
    num_sims = 2
    num_timesteps = 30
    h, w = 1024, 1024
    # Channels: fuel_class, DEM, slope, aspect, wind_u, wind_v, temp, humidity, precip, burn_mask
    num_channels = 10

    for i in range(num_sims):
        sim_dir = os.path.join(RAW_DATA_DIR, f'sim_{i}')
        os.makedirs(sim_dir, exist_ok=True)
        burn_mask = np.zeros((h,w), dtype=np.float32)
        for t in range(num_timesteps):
            dummy_raster = np.random.rand(num_channels, h, w).astype(np.float32)
            # Simulate burn spread
            if t > 0:
                # Simple cellular automata-like spread
                burn_mask_padded = np.pad(burn_mask, 1, 'constant')
                new_burn = (burn_mask_padded[:-2, 1:-1] + burn_mask_padded[2:, 1:-1] +
                            burn_mask_padded[1:-1, :-2] + burn_mask_padded[1:-1, 2:] > 0.5)
                burn_mask = np.clip(burn_mask + new_burn * 0.2 + np.random.rand(h,w) * 0.05, 0, 1)
            elif t==0:
                 # Initial ignition
                 burn_mask[h//2-10:h//2+10, w//2-10:w//2+10] = 1.0

            dummy_raster[-1,:,:] = burn_mask
            np.save(os.path.join(sim_dir, f'raster_{t:03d}.npy'), dummy_raster)

if __name__ == "__main__":
    convert_spread_dataset()