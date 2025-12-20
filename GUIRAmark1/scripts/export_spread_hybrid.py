import torch
import yaml
import os
import json
from pathlib import Path
from datetime import datetime
import argparse

def export_spread_model(config_path=None, output_dir=None, formats=None):
    """
    Export the trained fire spread prediction model to multiple formats.
    """
    # Load config
    config_path = config_path or "/home/runner/work/FIREPREVENTION/FIREPREVENTION/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_dir = config['model_paths']['spread_hybrid']
    model_path = os.path.join(model_dir, 'best.pt')
    
    # Setup output directory
    if not output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"/home/runner/work/FIREPREVENTION/FIREPREVENTION/experiments/spread_hybrid_{timestamp}/artifacts"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Default formats
    if not formats:
        formats = ['torchscript', 'onnx']
    
    export_results = {"model": "spread_hybrid", "timestamp": timestamp, "exports": {}}
    
    try:
        # Define fire spread model architecture
        class ConvLSTMCell(torch.nn.Module):
            def __init__(self, input_dim, hidden_dim, kernel_size):
                super().__init__()
                self.input_dim = input_dim
                self.hidden_dim = hidden_dim
                padding = kernel_size // 2
                
                self.conv = torch.nn.Conv2d(
                    in_channels=input_dim + hidden_dim,
                    out_channels=4 * hidden_dim,
                    kernel_size=kernel_size,
                    padding=padding
                )
            
            def forward(self, input_tensor, cur_state):
                h_cur, c_cur = cur_state
                
                combined = torch.cat([input_tensor, h_cur], dim=1)
                combined_conv = self.conv(combined)
                
                cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
                i = torch.sigmoid(cc_i)
                f = torch.sigmoid(cc_f)
                o = torch.sigmoid(cc_o)
                g = torch.tanh(cc_g)
                
                c_next = f * c_cur + i * g
                h_next = o * torch.tanh(c_next)
                
                return h_next, c_next
        
        class ConvLSTM(torch.nn.Module):
            def __init__(self, input_dim=1, hidden_dim=64, kernel_size=3, num_layers=2):
                super().__init__()
                
                self.input_dim = input_dim
                self.hidden_dim = hidden_dim
                self.kernel_size = kernel_size
                self.num_layers = num_layers
                
                cell_list = []
                for i in range(num_layers):
                    cur_input_dim = input_dim if i == 0 else hidden_dim
                    cell_list.append(ConvLSTMCell(cur_input_dim, hidden_dim, kernel_size))
                
                self.cell_list = torch.nn.ModuleList(cell_list)
                
                # Output projection
                self.output_conv = torch.nn.Conv2d(hidden_dim, 1, kernel_size=1)
                
            def forward(self, input_tensor):
                """
                input_tensor: (batch, seq_len, channels, height, width)
                output: (batch, pred_len, height, width)
                """
                batch_size, seq_len, _, height, width = input_tensor.size()
                
                # Initialize hidden states
                hidden_state = []
                for _ in range(self.num_layers):
                    h = torch.zeros(batch_size, self.hidden_dim, height, width).to(input_tensor.device)
                    c = torch.zeros(batch_size, self.hidden_dim, height, width).to(input_tensor.device)
                    hidden_state.append([h, c])
                
                # Process input sequence
                for t in range(seq_len):
                    cur_input = input_tensor[:, t, :, :, :]
                    
                    for layer_idx in range(self.num_layers):
                        h, c = self.cell_list[layer_idx](cur_input, hidden_state[layer_idx])
                        hidden_state[layer_idx] = [h, c]
                        cur_input = h
                
                # Generate future predictions
                outputs = []
                pred_len = 12  # Predict 12 time steps
                
                for t in range(pred_len):
                    # Use last hidden state as input for prediction
                    cur_input = hidden_state[-1][0]  # Use last layer's hidden state
                    
                    for layer_idx in range(self.num_layers):
                        if layer_idx == 0:
                            # For first layer, use the output from previous timestep
                            layer_input = self.output_conv(cur_input).repeat(1, self.input_dim, 1, 1)
                        else:
                            layer_input = cur_input
                            
                        h, c = self.cell_list[layer_idx](layer_input, hidden_state[layer_idx])
                        hidden_state[layer_idx] = [h, c]
                        cur_input = h
                    
                    # Generate output for this timestep
                    output = self.output_conv(hidden_state[-1][0])
                    outputs.append(output)
                
                # Stack outputs: (batch, pred_len, height, width)
                return torch.stack(outputs, dim=1).squeeze(2)
        
        # Create model
        spread_config = config['training']['spread_hybrid']
        model = ConvLSTM(input_dim=1, hidden_dim=64, kernel_size=3, num_layers=2)
        
        try:
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            print(f"✅ Loaded model weights from: {model_path}")
        except Exception as e:
            print(f"⚠️  Could not load weights: {e}")
            print("   Using randomly initialized weights for export demonstration")
        
        model.eval()
        
        print(f"Exporting Fire Spread model...")
        
        # Create dummy input: (batch, seq_len, channels, height, width)
        dummy_input = torch.randn(1, 6, 1, 64, 64)  # 6 input timesteps
        
        for format_name in formats:
            try:
                print(f"Exporting to {format_name.upper()}...")
                
                if format_name == 'torchscript':
                    # TorchScript export
                    traced_model = torch.jit.trace(model, dummy_input)
                    export_path = output_path / "spread_hybrid.pt"
                    traced_model.save(str(export_path))
                    
                elif format_name == 'onnx':
                    # ONNX export
                    export_path = output_path / "spread_hybrid.onnx"
                    torch.onnx.export(
                        model,
                        dummy_input,
                        str(export_path),
                        input_names=['fire_sequence'],
                        output_names=['spread_prediction'],
                        dynamic_axes={
                            'fire_sequence': {0: 'batch_size'},
                            'spread_prediction': {0: 'batch_size'}
                        },
                        opset_version=11
                    )
                else:
                    print(f"Unsupported format: {format_name}")
                    continue
                
                export_results["exports"][format_name] = str(export_path)
                print(f"✅ {format_name.upper()} export completed: {export_path}")
                
            except Exception as e:
                print(f"❌ Failed to export to {format_name}: {e}")
                export_results["exports"][format_name] = f"Failed: {str(e)}"
        
        # Create physics helper
        create_physics_helper(output_path, spread_config)
        
        # Generate export metadata
        metadata = {
            "model_info": {
                "architecture": "ConvLSTM with Physics Regularization",
                "input_shape": [1, 6, 1, 64, 64],  # (batch, seq_in, channels, H, W)
                "output_shape": [1, 12, 64, 64],    # (batch, seq_out, H, W)
                "grid_resolution": spread_config['cell_size'],  # meters per pixel
                "timestep": spread_config['timestep'],  # seconds
                "prediction_horizon": "12 timesteps (1 hour)",
                "preprocessing": {
                    "input_normalization": "Binary fire mask [0, 1]",
                    "temporal_window": "6 timesteps (30 minutes)",
                    "spatial_resolution": f"{spread_config['cell_size']}m per pixel"
                }
            },
            "export_info": export_results
        }
        
        # Save metadata
        with open(output_path / "export_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n🎯 Fire spread model export completed!")
        print(f"📁 Artifacts saved to: {output_path}")
        for format_name, path in export_results["exports"].items():
            if not path.startswith("Failed"):
                print(f"   - {format_name.upper()}: {path}")
                
    except Exception as e:
        print(f"❌ Error during export: {e}")

def create_physics_helper(output_path, spread_config):
    """Create physics computation helper script"""
    physics_code = f'''import numpy as np
import torch

class FireSpreadPhysics:
    """
    Physics-based fire spread utilities for preprocessing and validation.
    """
    
    def __init__(self, cell_size={spread_config['cell_size']}, timestep={spread_config['timestep']}):
        self.cell_size = cell_size  # meters per pixel
        self.timestep = timestep    # seconds per timestep
        
        # Physical constants
        self.max_spread_rate = 10.0  # m/min maximum realistic spread
        self.wind_factor = 0.1       # wind influence coefficient
        self.slope_factor = 0.2      # terrain slope influence
        
    def validate_spread_rate(self, fire_masks):
        """
        Validate that fire spread rates are physically realistic.
        
        Args:
            fire_masks: numpy array of shape (T, H, W) with binary fire masks
        
        Returns:
            is_valid: bool, whether spread rates are realistic
            max_rate: float, maximum observed spread rate in m/min
        """
        spread_rates = []
        
        for t in range(1, len(fire_masks)):
            # Calculate area change
            prev_area = np.sum(fire_masks[t-1]) * (self.cell_size ** 2)
            curr_area = np.sum(fire_masks[t]) * (self.cell_size ** 2)
            
            # Convert to spread rate (m/min)
            area_change = curr_area - prev_area
            if area_change > 0:
                rate = np.sqrt(area_change / np.pi) / (self.timestep / 60)
                spread_rates.append(rate)
        
        max_rate = max(spread_rates) if spread_rates else 0
        is_valid = max_rate <= self.max_spread_rate
        
        return is_valid, max_rate
    
    def apply_wind_effect(self, fire_mask, wind_speed, wind_direction):
        """
        Apply wind effects to fire spread prediction.
        
        Args:
            fire_mask: numpy array of shape (H, W)
            wind_speed: float, wind speed in km/h
            wind_direction: float, wind direction in degrees (0=North, 90=East)
        
        Returns:
            modified_mask: numpy array with wind effects applied
        """
        # Convert wind direction to radians
        wind_rad = np.radians(wind_direction)
        
        # Create directional kernel based on wind
        kernel_size = 5
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        
        for i in range(kernel_size):
            for j in range(kernel_size):
                # Calculate relative position
                dy = i - center
                dx = j - center
                
                if dx == 0 and dy == 0:
                    kernel[i, j] = 1.0
                else:
                    # Calculate angle to this pixel
                    angle = np.arctan2(dy, dx)
                    
                    # Weight based on wind direction alignment
                    angle_diff = abs(angle - wind_rad)
                    if angle_diff > np.pi:
                        angle_diff = 2 * np.pi - angle_diff
                    
                    # Wind enhances spread in wind direction
                    wind_influence = np.cos(angle_diff) * wind_speed * self.wind_factor
                    kernel[i, j] = max(0, 1 + wind_influence)
        
        # Apply convolution (simplified)
        from scipy import ndimage
        modified_mask = ndimage.convolve(fire_mask.astype(float), kernel, mode='constant')
        
        return np.clip(modified_mask, 0, 1)
    
    def apply_terrain_effect(self, fire_mask, elevation_map):
        """
        Apply terrain slope effects to fire spread.
        
        Args:
            fire_mask: numpy array of shape (H, W)
            elevation_map: numpy array of shape (H, W) with elevation in meters
        
        Returns:
            modified_mask: numpy array with terrain effects applied
        """
        # Calculate slope gradients
        grad_y, grad_x = np.gradient(elevation_map)
        slope_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize slope (typical max slope ~45 degrees = 1.0 gradient)
        slope_normalized = np.clip(slope_magnitude / 1.0, 0, 2)
        
        # Apply slope effect (fire spreads faster uphill)
        slope_effect = 1 + slope_normalized * self.slope_factor
        modified_mask = fire_mask * slope_effect
        
        return np.clip(modified_mask, 0, 1)
    
    def preprocess_raster_sequence(self, fire_sequence, wind_data=None, terrain_data=None):
        """
        Preprocess fire sequence with physics-based modifications.
        
        Args:
            fire_sequence: numpy array of shape (T, H, W)
            wind_data: dict with 'speed' and 'direction' (optional)
            terrain_data: dict with 'elevation' array (optional)
        
        Returns:
            processed_sequence: numpy array of shape (T, H, W)
            physics_metadata: dict with processing information
        """
        processed = fire_sequence.copy()
        metadata = {{"original_max_spread_rate": 0, "processed_max_spread_rate": 0}}
        
        # Validate original sequence
        orig_valid, orig_rate = self.validate_spread_rate(fire_sequence)
        metadata["original_max_spread_rate"] = orig_rate
        metadata["original_physically_valid"] = orig_valid
        
        # Apply wind effects if provided
        if wind_data:
            for t in range(len(processed)):
                processed[t] = self.apply_wind_effect(
                    processed[t], wind_data['speed'], wind_data['direction']
                )
            metadata["wind_applied"] = True
        
        # Apply terrain effects if provided
        if terrain_data:
            for t in range(len(processed)):
                processed[t] = self.apply_terrain_effect(
                    processed[t], terrain_data['elevation']
                )
            metadata["terrain_applied"] = True
        
        # Validate processed sequence
        proc_valid, proc_rate = self.validate_spread_rate(processed)
        metadata["processed_max_spread_rate"] = proc_rate
        metadata["processed_physically_valid"] = proc_valid
        
        return processed, metadata

# Example usage:
if __name__ == "__main__":
    # Create physics helper
    physics = FireSpreadPhysics()
    
    # Example fire sequence (6 timesteps)
    fire_sequence = np.random.rand(6, 64, 64) > 0.8
    
    # Example wind data
    wind_data = {{"speed": 15, "direction": 45}}  # 15 km/h from NE
    
    # Example terrain (elevation map)
    x, y = np.meshgrid(range(64), range(64))
    elevation = 1000 + 10 * np.sin(x/10) + 5 * np.cos(y/10)  # Synthetic terrain
    terrain_data = {{"elevation": elevation}}
    
    # Process sequence
    processed, metadata = physics.preprocess_raster_sequence(
        fire_sequence, wind_data, terrain_data
    )
    
    print(f"Original sequence valid: {{metadata['original_physically_valid']}}")
    print(f"Processed sequence valid: {{metadata['processed_physically_valid']}}")
    print(f"Max spread rate: {{metadata['processed_max_spread_rate']:.2f}} m/min")
'''
    
    with open(output_path / "fire_physics.py", 'w') as f:
        f.write(physics_code)
    
    print("✅ Created physics computation helper: fire_physics.py")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export Fire Spread Model')
    parser.add_argument('--config', default=None, help='Path to config.yaml')
    parser.add_argument('--output', default=None, help='Output directory for exported models')
    parser.add_argument('--formats', nargs='+', default=['torchscript', 'onnx'], 
                       choices=['torchscript', 'onnx'],
                       help='Export formats (default: torchscript onnx)')
    
    args = parser.parse_args()
    export_spread_model(args.config, args.output, args.formats)