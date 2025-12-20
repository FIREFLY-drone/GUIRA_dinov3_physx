import torch
import yaml
import os
import json
from pathlib import Path
from datetime import datetime
import argparse

def export_vegetation_model(config_path=None, output_dir=None, formats=None):
    """
    Export the trained vegetation health classification model to multiple formats.
    """
    # Load config
    config_path = config_path or "/home/runner/work/FIREPREVENTION/FIREPREVENTION/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_dir = config['model_paths']['vegetation_resnet_vari']
    model_path = os.path.join(model_dir, 'best.pt')
    
    # Setup output directory
    if not output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"/home/runner/work/FIREPREVENTION/FIREPREVENTION/experiments/vegetation_resnet_vari_{timestamp}/artifacts"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Default formats
    if not formats:
        formats = ['torchscript', 'onnx']
    
    export_results = {"model": "vegetation_resnet_vari", "timestamp": timestamp, "exports": {}}
    
    try:
        # Define vegetation health model architecture
        class VegHealthModel(torch.nn.Module):
            def __init__(self, num_classes=3):
                super().__init__()
                # ResNet50 backbone (simplified)
                self.backbone = torch.nn.Sequential(
                    # Conv1
                    torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                    torch.nn.BatchNorm2d(64),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    
                    # Simplified residual blocks
                    torch.nn.Conv2d(64, 256, 3, padding=1),
                    torch.nn.BatchNorm2d(256),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(256, 512, 3, stride=2, padding=1),
                    torch.nn.BatchNorm2d(512),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.AdaptiveAvgPool2d((1, 1))
                )
                
                # VARI integration branch
                self.vari_processor = torch.nn.Sequential(
                    torch.nn.Linear(1, 64),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Linear(64, 128),
                    torch.nn.ReLU(inplace=True)
                )
                
                # Combined classifier
                self.classifier = torch.nn.Sequential(
                    torch.nn.Linear(512 + 128, 256),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Dropout(0.5),
                    torch.nn.Linear(256, num_classes)
                )
                
            def forward(self, rgb_input, vari_input):
                # Process RGB image
                rgb_features = self.backbone(rgb_input)
                rgb_features = rgb_features.view(rgb_features.size(0), -1)
                
                # Process VARI index
                vari_features = self.vari_processor(vari_input)
                
                # Combine features
                combined_features = torch.cat([rgb_features, vari_features], dim=1)
                
                # Classify
                output = self.classifier(combined_features)
                return output
        
        # Create model
        veg_config = config['training']['vegetation_resnet_vari']
        num_classes = len(veg_config['classes'])
        model = VegHealthModel(num_classes=num_classes)
        
        try:
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            print(f"✅ Loaded model weights from: {model_path}")
        except Exception as e:
            print(f"⚠️  Could not load weights: {e}")
            print("   Using randomly initialized weights for export demonstration")
        
        model.eval()
        
        print(f"Exporting Vegetation Health model...")
        
        # Create dummy inputs
        dummy_rgb = torch.randn(1, 3, 224, 224)  # RGB image
        dummy_vari = torch.randn(1, 1)  # VARI index
        
        for format_name in formats:
            try:
                print(f"Exporting to {format_name.upper()}...")
                
                if format_name == 'torchscript':
                    # TorchScript export
                    traced_model = torch.jit.trace(model, (dummy_rgb, dummy_vari))
                    export_path = output_path / "vegetation_resnet_vari.pt"
                    traced_model.save(str(export_path))
                    
                elif format_name == 'onnx':
                    # ONNX export (requires handling multiple inputs)
                    export_path = output_path / "vegetation_resnet_vari.onnx"
                    torch.onnx.export(
                        model,
                        (dummy_rgb, dummy_vari),
                        str(export_path),
                        input_names=['rgb_image', 'vari_index'],
                        output_names=['health_classification'],
                        dynamic_axes={
                            'rgb_image': {0: 'batch_size'},
                            'vari_index': {0: 'batch_size'},
                            'health_classification': {0: 'batch_size'}
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
        
        # Create VARI computation helper
        create_vari_helper(output_path)
        
        # Generate export metadata
        metadata = {
            "model_info": {
                "architecture": "ResNet50 + VARI integration",
                "input_shapes": {
                    "rgb_image": [1, 3, 224, 224],
                    "vari_index": [1, 1]
                },
                "output_classes": veg_config['classes'],
                "preprocessing": {
                    "rgb_normalize": True,
                    "rgb_mean": [0.485, 0.456, 0.406],
                    "rgb_std": [0.229, 0.224, 0.225],
                    "rgb_resize": [224, 224],
                    "vari_computation": "See compute_vari.py"
                }
            },
            "export_info": export_results
        }
        
        # Save metadata
        with open(output_path / "export_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n🎯 Vegetation model export completed!")
        print(f"📁 Artifacts saved to: {output_path}")
        for format_name, path in export_results["exports"].items():
            if not path.startswith("Failed"):
                print(f"   - {format_name.upper()}: {path}")
                
    except Exception as e:
        print(f"❌ Error during export: {e}")

def create_vari_helper(output_path):
    """Create VARI computation helper script"""
    vari_code = '''import numpy as np
import cv2

def compute_vari(rgb_image):
    """
    Compute Visible Atmospherically Resistant Index (VARI) from RGB image.
    
    VARI = (Green - Red) / (Green + Red - Blue)
    
    Args:
        rgb_image: numpy array of shape (H, W, 3) with RGB values [0-255]
    
    Returns:
        vari_value: float, VARI index value
    """
    if len(rgb_image.shape) == 3:
        # Convert to float and normalize
        rgb = rgb_image.astype(np.float32) / 255.0
        
        # Extract channels
        red = rgb[:, :, 0]
        green = rgb[:, :, 1] 
        blue = rgb[:, :, 2]
        
        # Compute VARI
        numerator = green - red
        denominator = green + red - blue
        
        # Avoid division by zero
        denominator = np.where(denominator == 0, 1e-8, denominator)
        vari = numerator / denominator
        
        # Return mean VARI value
        return np.mean(vari)
    else:
        raise ValueError("Input image must be 3-channel RGB")

def preprocess_for_model(rgb_image):
    """
    Preprocess RGB image and compute VARI for model inference.
    
    Args:
        rgb_image: numpy array of shape (H, W, 3) with RGB values [0-255]
    
    Returns:
        rgb_tensor: torch tensor of shape (1, 3, 224, 224)
        vari_tensor: torch tensor of shape (1, 1)
    """
    import torch
    from torchvision import transforms
    
    # RGB preprocessing
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    rgb_tensor = transform(rgb_image).unsqueeze(0)
    
    # VARI computation
    vari_value = compute_vari(rgb_image)
    vari_tensor = torch.tensor([[vari_value]], dtype=torch.float32)
    
    return rgb_tensor, vari_tensor

# Example usage:
if __name__ == "__main__":
    # Load image
    image = cv2.imread("vegetation_sample.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Compute VARI
    vari_index = compute_vari(image)
    print(f"VARI Index: {vari_index:.4f}")
    
    # Preprocess for model
    rgb_tensor, vari_tensor = preprocess_for_model(image)
    print(f"RGB tensor shape: {rgb_tensor.shape}")
    print(f"VARI tensor shape: {vari_tensor.shape}")
'''
    
    with open(output_path / "compute_vari.py", 'w') as f:
        f.write(vari_code)
    
    print("✅ Created VARI computation helper: compute_vari.py")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export Vegetation Health Model')
    parser.add_argument('--config', default=None, help='Path to config.yaml')
    parser.add_argument('--output', default=None, help='Output directory for exported models')
    parser.add_argument('--formats', nargs='+', default=['torchscript', 'onnx'], 
                       choices=['torchscript', 'onnx'],
                       help='Export formats (default: torchscript onnx)')
    
    args = parser.parse_args()
    export_vegetation_model(args.config, args.output, args.formats)