import yaml
from ultralytics import YOLO
import os
from pathlib import Path
import torch
import argparse
from datetime import datetime

def export_fire_model(config_path=None, output_dir=None, formats=None):
    """
    Export the trained YOLOv8 fire detection model to multiple formats.
    Supports ONNX, TorchScript, and TensorRT formats.
    """
    # Load config
    config_path = config_path or "/home/runner/work/FIREPREVENTION/FIREPREVENTION/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_path = os.path.join(config['model_paths']['fire_yolov8'], 'best.pt')
    
    # Setup output directory
    if not output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"/home/runner/work/FIREPREVENTION/FIREPREVENTION/experiments/fire_yolov8_{timestamp}/artifacts"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Default formats
    if not formats:
        formats = ['onnx', 'torchscript']
    
    export_results = {"model": "fire_yolov8", "timestamp": timestamp, "exports": {}}

    try:
        # Load the trained model
        model = YOLO(model_path)
        
        print(f"Exporting Fire Detection model from: {model_path}")
        
        for format_name in formats:
            try:
                print(f"Exporting to {format_name.upper()}...")
                
                if format_name == 'onnx':
                    export_path = model.export(format='onnx', dynamic=True, simplify=True)
                elif format_name == 'torchscript':
                    export_path = model.export(format='torchscript')
                elif format_name == 'tensorrt':
                    export_path = model.export(format='engine', device=0)  # Requires GPU
                else:
                    print(f"Unsupported format: {format_name}")
                    continue
                
                export_results["exports"][format_name] = str(export_path)
                print(f"✅ {format_name.upper()} export completed: {export_path}")
                
                # Move to organized location
                if export_path and os.path.exists(export_path):
                    target_path = output_path / f"fire_yolov8.{format_name.replace('torchscript', 'pt')}"
                    if format_name == 'onnx':
                        target_path = output_path / "fire_yolov8.onnx"
                    elif format_name == 'tensorrt':
                        target_path = output_path / "fire_yolov8.engine"
                    
                    os.rename(export_path, target_path)
                    export_results["exports"][format_name] = str(target_path)
                    
            except Exception as e:
                print(f"❌ Failed to export to {format_name}: {e}")
                export_results["exports"][format_name] = f"Failed: {str(e)}"
        
        # Generate export metadata
        metadata = {
            "model_info": {
                "architecture": "YOLOv8n",
                "input_shape": [1, 3, 640, 640],
                "output_classes": ["fire", "smoke"],
                "preprocessing": {
                    "normalize": True,
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225]
                }
            },
            "export_info": export_results
        }
        
        # Save metadata
        import json
        with open(output_path / "export_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n🎯 Fire model export completed!")
        print(f"📁 Artifacts saved to: {output_path}")
        for format_name, path in export_results["exports"].items():
            if not path.startswith("Failed"):
                print(f"   - {format_name.upper()}: {path}")
                
    except Exception as e:
        print(f"❌ Error loading model from {model_path}: {e}")
        print("   Make sure the model file exists and is trained properly.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export Fire Detection Model')
    parser.add_argument('--config', default=None, help='Path to config.yaml')
    parser.add_argument('--output', default=None, help='Output directory for exported models')
    parser.add_argument('--formats', nargs='+', default=['onnx', 'torchscript'], 
                       choices=['onnx', 'torchscript', 'tensorrt'],
                       help='Export formats (default: onnx torchscript)')
    
    args = parser.parse_args()
    export_fire_model(args.config, args.output, args.formats)
