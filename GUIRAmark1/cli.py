"""
Command Line Interface for Fire Prevention System
Provides unified CLI access to all system components.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, Optional
import subprocess

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from utils import setup_logging, load_config
from loguru import logger


class FirePreventionCLI:
    """Command line interface for the fire prevention system."""
    
    def __init__(self):
        self.config = None
        self.scripts_dir = Path(__file__).parent
    
    def load_config(self, config_path: str = 'config.yaml'):
        """Load configuration file."""
        try:
            self.config = load_config(config_path)
            logger.info(f"Loaded configuration from {config_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return False
    
    def train_models(self, model_type: str = 'all', **kwargs):
        """Train models."""
        logger.info(f"Training {model_type} models...")
        
        training_scripts = {
            'fire': 'train_fire.py',
            'smoke': 'train_smoke.py',
            'fauna': 'train_fauna.py',
            'vegetation': 'train_veg.py',
            'spread': 'simulate_spread.py'
        }
        
        if model_type == 'all':
            scripts_to_run = list(training_scripts.values())
        else:
            if model_type not in training_scripts:
                logger.error(f"Unknown model type: {model_type}")
                return False
            scripts_to_run = [training_scripts[model_type]]
        
        success = True
        for script in scripts_to_run:
            script_path = self.scripts_dir / script
            if not script_path.exists():
                logger.error(f"Training script not found: {script}")
                success = False
                continue
            
            logger.info(f"Running {script}...")
            cmd = [sys.executable, str(script_path)]
            
            # Add command line arguments
            if 'epochs' in kwargs:
                cmd.extend(['--epochs', str(kwargs['epochs'])])
            if 'batch_size' in kwargs:
                cmd.extend(['--batch_size', str(kwargs['batch_size'])])
            if 'lr' in kwargs:
                cmd.extend(['--lr', str(kwargs['lr'])])
            if 'config' in kwargs:
                cmd.extend(['--config', kwargs['config']])
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info(f"✅ {script} completed successfully")
                else:
                    logger.error(f"❌ {script} failed:")
                    logger.error(result.stderr)
                    success = False
            except Exception as e:
                logger.error(f"Failed to run {script}: {e}")
                success = False
        
        return success
    
    def run_inference(self, model_type: str, input_path: str, output_path: str, **kwargs):
        """Run inference on input data."""
        logger.info(f"Running {model_type} inference...")
        
        inference_commands = {
            'fire': self._run_fire_inference,
            'smoke': self._run_smoke_inference,
            'fauna': self._run_fauna_inference,
            'vegetation': self._run_vegetation_inference,
            'spread': self._run_spread_simulation
        }
        
        if model_type not in inference_commands:
            logger.error(f"Unknown model type: {model_type}")
            return False
        
        return inference_commands[model_type](input_path, output_path, **kwargs)
    
    def _run_fire_inference(self, input_path: str, output_path: str, **kwargs):
        """Run fire detection inference."""
        try:
            from fire.fire_detection import FireDetectionInference
            
            if not self.config:
                logger.error("Configuration not loaded")
                return False
            
            model_path = kwargs.get('model_path', self.config['fire']['model_path'])
            detector = FireDetectionInference(model_path, self.config['fire'])
            
            # Process input
            import cv2
            image = cv2.imread(input_path)
            if image is None:
                logger.error(f"Could not load image: {input_path}")
                return False
            
            # Run detection
            detections = detector.detect(image)
            
            # Save results
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(output_dir / 'detections.json', 'w') as f:
                json.dump(detections, f, indent=2)
            
            logger.info(f"Fire detection results saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Fire inference failed: {e}")
            return False
    
    def _run_smoke_inference(self, input_path: str, output_path: str, **kwargs):
        """Run smoke detection inference."""
        try:
            from smoke.smoke_detection import SmokeDetectionInference
            
            if not self.config:
                logger.error("Configuration not loaded")
                return False
            
            model_path = kwargs.get('model_path', self.config['smoke']['model_path'])
            detector = SmokeDetectionInference(model_path, self.config['smoke'])
            
            # Process video
            import cv2
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                logger.error(f"Could not open video: {input_path}")
                return False
            
            frames = []
            sequence_length = self.config['smoke']['sequence_length']
            
            while len(frames) < sequence_length:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            cap.release()
            
            if len(frames) < sequence_length:
                logger.error(f"Not enough frames for smoke detection (need {sequence_length})")
                return False
            
            # Run detection
            result = detector.detect_sequence(frames)
            
            # Save results
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(output_dir / 'smoke_result.json', 'w') as f:
                json.dump(result, f, indent=2)
            
            logger.info(f"Smoke detection results saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Smoke inference failed: {e}")
            return False
    
    def _run_fauna_inference(self, input_path: str, output_path: str, **kwargs):
        """Run fauna detection inference."""
        try:
            from fauna.fauna_detection import FaunaDetectionInference
            
            if not self.config:
                logger.error("Configuration not loaded")
                return False
            
            model_path = kwargs.get('model_path', self.config['fauna']['model_path'])
            detector = FaunaDetectionInference(model_path, self.config['fauna'])
            
            # Process input
            import cv2
            image = cv2.imread(input_path)
            if image is None:
                logger.error(f"Could not load image: {input_path}")
                return False
            
            # Run detection
            detections, density_map = detector.detect(image)
            
            # Save results
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(output_dir / 'fauna_detections.json', 'w') as f:
                json.dump(detections, f, indent=2)
            
            # Save density map
            import numpy as np
            np.save(output_dir / 'density_map.npy', density_map)
            
            logger.info(f"Fauna detection results saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Fauna inference failed: {e}")
            return False
    
    def _run_vegetation_inference(self, input_path: str, output_path: str, **kwargs):
        """Run vegetation health inference."""
        try:
            from vegetation.vegetation_health import VegetationHealthInference
            
            if not self.config:
                logger.error("Configuration not loaded")
                return False
            
            model_path = kwargs.get('model_path', self.config['vegetation']['model_path'])
            detector = VegetationHealthInference(model_path, self.config['vegetation'])
            
            # Process input
            import cv2
            image = cv2.imread(input_path)
            if image is None:
                logger.error(f"Could not load image: {input_path}")
                return False
            
            # Run analysis
            result = detector.analyze(image)
            
            # Save results
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(output_dir / 'vegetation_analysis.json', 'w') as f:
                json.dump(result, f, indent=2)
            
            # Save health map if available
            if 'health_map' in result:
                import cv2
                health_map = result['health_map']
                cv2.imwrite(str(output_dir / 'health_map.png'), health_map)
            
            logger.info(f"Vegetation analysis results saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Vegetation inference failed: {e}")
            return False
    
    def _run_spread_simulation(self, input_path: str, output_path: str, **kwargs):
        """Run fire spread simulation."""
        try:
            from spread.fire_spread_simulation import FireSpreadSimulator
            
            if not self.config:
                logger.error("Configuration not loaded")
                return False
            
            model_path = kwargs.get('model_path', self.config['spread']['model_path'])
            simulator = FireSpreadSimulator(model_path, self.config['spread'])
            
            # Load input parameters
            with open(input_path, 'r') as f:
                sim_params = json.load(f)
            
            # Create initial fire state
            grid_size = self.config['spread']['grid_size']
            initial_state = simulator.create_initial_fire_state(
                sim_params.get('ignition_points', [(64, 64)])
            )
            
            # Create environmental state
            env_state = simulator.create_environmental_state(
                wind_speed=sim_params.get('wind_speed', 10.0),
                wind_direction=sim_params.get('wind_direction', 45.0),
                humidity=sim_params.get('humidity', 0.3),
                vegetation_density=sim_params.get('vegetation_density', 0.8)
            )
            
            # Run simulation
            steps = kwargs.get('steps', 50)
            fire_states = simulator.simulate(initial_state, env_state, steps=steps)
            
            # Save results
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            simulator.save_simulation_results(fire_states, str(output_dir))
            simulator.visualize_simulation(fire_states, str(output_dir / 'simulation.gif'))
            
            logger.info(f"Fire spread simulation results saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Fire spread simulation failed: {e}")
            return False
    
    def run_pipeline(self, input_source: str, **kwargs):
        """Run the complete pipeline."""
        logger.info("Running complete fire prevention pipeline...")
        
        pipeline_script = self.scripts_dir / 'run_pipeline.py'
        if not pipeline_script.exists():
            logger.error("Pipeline script not found")
            return False
        
        cmd = [sys.executable, str(pipeline_script)]
        
        # Add arguments
        if input_source:
            cmd.extend(['--input', input_source])
        if 'max_frames' in kwargs:
            cmd.extend(['--max_frames', str(kwargs['max_frames'])])
        if 'output_dir' in kwargs:
            cmd.extend(['--output_dir', kwargs['output_dir']])
        if 'config' in kwargs:
            cmd.extend(['--config', kwargs['config']])
        
        try:
            result = subprocess.run(cmd, capture_output=False, text=True)
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            return False
    
    def download_datasets(self, dataset_type: str = 'all'):
        """Download datasets."""
        logger.info(f"Downloading {dataset_type} datasets...")
        
        download_script = self.scripts_dir / 'download_datasets.py'
        if not download_script.exists():
            logger.error("Dataset download script not found")
            return False
        
        cmd = [sys.executable, str(download_script), '--dataset', dataset_type]
        
        try:
            result = subprocess.run(cmd, capture_output=False, text=True)
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Dataset download failed: {e}")
            return False
    
    def test_system(self, test_type: str = 'all'):
        """Run system tests."""
        logger.info(f"Running {test_type} tests...")
        
        test_script = self.scripts_dir / 'test_system.py'
        if not test_script.exists():
            logger.error("Test script not found")
            return False
        
        cmd = [sys.executable, str(test_script), '--test-type', test_type]
        
        try:
            result = subprocess.run(cmd, capture_output=False, text=True)
            return result.returncode == 0
        except Exception as e:
            logger.error(f"System testing failed: {e}")
            return False
    
    def show_status(self):
        """Show system status."""
        logger.info("Fire Prevention System Status")
        logger.info("=" * 40)
        
        # Check configuration
        if self.config:
            logger.info("✅ Configuration: Loaded")
        else:
            logger.info("❌ Configuration: Not loaded")
        
        # Check models directory
        models_dir = Path('models')
        if models_dir.exists():
            model_files = list(models_dir.glob('*.pt'))
            logger.info(f"📁 Models: {len(model_files)} files found")
            for model_file in model_files:
                size_mb = model_file.stat().st_size / (1024 * 1024)
                logger.info(f"   - {model_file.name} ({size_mb:.1f} MB)")
        else:
            logger.info("❌ Models: Directory not found")
        
        # Check data directory
        data_dir = Path('data')
        if data_dir.exists():
            subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
            logger.info(f"📁 Data: {len(subdirs)} subdirectories")
            for subdir in subdirs:
                file_count = len(list(subdir.rglob('*')))
                logger.info(f"   - {subdir.name}: {file_count} files")
        else:
            logger.info("❌ Data: Directory not found")
        
        # Check outputs directory
        outputs_dir = Path('outputs')
        if outputs_dir.exists():
            logger.info("✅ Outputs: Directory exists")
        else:
            logger.info("⚠️  Outputs: Directory will be created when needed")
        
        # System requirements
        try:
            import torch
            logger.info(f"✅ PyTorch: {torch.__version__}")
            if torch.cuda.is_available():
                logger.info(f"✅ CUDA: Available ({torch.cuda.device_count()} devices)")
            else:
                logger.info("⚠️  CUDA: Not available (CPU only)")
        except ImportError:
            logger.info("❌ PyTorch: Not installed")
        
        try:
            import cv2
            logger.info(f"✅ OpenCV: {cv2.__version__}")
        except ImportError:
            logger.info("❌ OpenCV: Not installed")


def main():
    # Create main parser
    parser = argparse.ArgumentParser(
        description='Fire Prevention System CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show system status
  python cli.py status
  
  # Download datasets
  python cli.py download --dataset all
  
  # Train all models
  python cli.py train --model all --epochs 50
  
  # Run inference
  python cli.py infer fire --input image.jpg --output results/
  
  # Run complete pipeline
  python cli.py pipeline --input 0 --max-frames 1000
  
  # Test system
  python cli.py test --type all
        """
    )
    
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    # Create subparsers
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show system status')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download datasets')
    download_parser.add_argument('--dataset', choices=['fire', 'smoke', 'fauna', 'vegetation', 'all'],
                                default='all', help='Dataset to download')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--model', choices=['fire', 'smoke', 'fauna', 'vegetation', 'spread', 'all'],
                             default='all', help='Model to train')
    train_parser.add_argument('--epochs', type=int, help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, help='Batch size')
    train_parser.add_argument('--lr', type=float, help='Learning rate')
    
    # Inference command
    infer_parser = subparsers.add_parser('infer', help='Run inference')
    infer_parser.add_argument('model', choices=['fire', 'smoke', 'fauna', 'vegetation', 'spread'],
                             help='Model to use for inference')
    infer_parser.add_argument('--input', required=True, help='Input file path')
    infer_parser.add_argument('--output', required=True, help='Output directory path')
    infer_parser.add_argument('--model-path', help='Override model path')
    infer_parser.add_argument('--steps', type=int, default=50, help='Simulation steps (for spread model)')
    
    # Pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run complete pipeline')
    pipeline_parser.add_argument('--input', help='Input source (camera index, video file, or stream URL)')
    pipeline_parser.add_argument('--max-frames', type=int, help='Maximum frames to process')
    pipeline_parser.add_argument('--output-dir', help='Output directory')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run system tests')
    test_parser.add_argument('--type', choices=['unit', 'integration', 'performance', 'all'],
                            default='all', help='Type of tests to run')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    setup_logging(log_level)
    
    # Initialize CLI
    cli = FirePreventionCLI()
    
    # Load configuration
    if not cli.load_config(args.config):
        logger.error("Failed to load configuration")
        sys.exit(1)
    
    # Execute command
    success = True
    
    if args.command == 'status':
        cli.show_status()
    
    elif args.command == 'download':
        success = cli.download_datasets(args.dataset)
    
    elif args.command == 'train':
        train_kwargs = {}
        if args.epochs:
            train_kwargs['epochs'] = args.epochs
        if args.batch_size:
            train_kwargs['batch_size'] = args.batch_size
        if args.lr:
            train_kwargs['lr'] = args.lr
        train_kwargs['config'] = args.config
        
        success = cli.train_models(args.model, **train_kwargs)
    
    elif args.command == 'infer':
        infer_kwargs = {}
        if args.model_path:
            infer_kwargs['model_path'] = args.model_path
        if args.steps:
            infer_kwargs['steps'] = args.steps
        
        success = cli.run_inference(args.model, args.input, args.output, **infer_kwargs)
    
    elif args.command == 'pipeline':
        pipeline_kwargs = {}
        if args.max_frames:
            pipeline_kwargs['max_frames'] = args.max_frames
        if args.output_dir:
            pipeline_kwargs['output_dir'] = args.output_dir
        pipeline_kwargs['config'] = args.config
        
        success = cli.run_pipeline(args.input, **pipeline_kwargs)
    
    elif args.command == 'test':
        success = cli.test_system(args.type)
    
    else:
        parser.print_help()
        sys.exit(1)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
