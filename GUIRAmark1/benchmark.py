"""
Performance Benchmarking Suite for Fire Prevention System
Measures and analyzes system performance across different scenarios.
"""

import argparse
import time
import psutil
import os
import sys
import json
import numpy as np
import cv2
import torch
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from dataclasses import dataclass
from collections import defaultdict

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from utils import setup_logging, load_config
from loguru import logger


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    execution_time: float
    memory_usage: float
    throughput: float
    accuracy: float = 0.0
    additional_metrics: Dict = None
    
    def __post_init__(self):
        if self.additional_metrics is None:
            self.additional_metrics = {}


class PerformanceBenchmark:
    """Performance benchmarking utilities."""
    
    def __init__(self, config_path: str = 'config.yaml'):
        self.config = load_config(config_path)
        self.results = []
        self.process = psutil.Process(os.getpid())
        
    def measure_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def benchmark_function(self, func, *args, **kwargs) -> BenchmarkResult:
        """Benchmark a function execution."""
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Measure initial memory
        initial_memory = self.measure_memory_usage()
        
        # Warm up
        try:
            func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")
        
        # Actual benchmark
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            logger.error(f"Benchmark function failed: {e}")
            result = None
            success = False
        
        end_time = time.time()
        
        # Measure final memory
        final_memory = self.measure_memory_usage()
        
        execution_time = end_time - start_time
        memory_usage = final_memory - initial_memory
        
        return BenchmarkResult(
            name=func.__name__,
            execution_time=execution_time,
            memory_usage=memory_usage,
            throughput=1.0 / execution_time if execution_time > 0 else 0.0,
            additional_metrics={'success': success, 'result': result}
        )
    
    def benchmark_image_processing(self, image_sizes: List[Tuple[int, int]]) -> List[BenchmarkResult]:
        """Benchmark image processing operations."""
        logger.info("Benchmarking image processing...")
        
        results = []
        
        for width, height in image_sizes:
            logger.info(f"Testing image size: {width}x{height}")
            
            # Create test image
            test_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            
            # Benchmark resize operation
            def resize_operation():
                return cv2.resize(test_image, (640, 640))
            
            result = self.benchmark_function(resize_operation)
            result.name = f"resize_{width}x{height}"
            results.append(result)
            
            # Benchmark normalization
            def normalize_operation():
                return test_image.astype(np.float32) / 255.0
            
            result = self.benchmark_function(normalize_operation)
            result.name = f"normalize_{width}x{height}"
            results.append(result)
            
            # Benchmark blur operation
            def blur_operation():
                return cv2.GaussianBlur(test_image, (15, 15), 0)
            
            result = self.benchmark_function(blur_operation)
            result.name = f"blur_{width}x{height}"
            results.append(result)
        
        return results
    
    def benchmark_model_inference(self) -> List[BenchmarkResult]:
        """Benchmark model inference speed."""
        logger.info("Benchmarking model inference...")
        
        results = []
        
        # Test different batch sizes
        batch_sizes = [1, 4, 8, 16]
        image_size = (640, 640)
        
        for batch_size in batch_sizes:
            logger.info(f"Testing batch size: {batch_size}")
            
            # Create test batch
            test_batch = np.random.randint(
                0, 255, (batch_size, image_size[1], image_size[0], 3), dtype=np.uint8
            )
            
            # Benchmark tensor conversion
            def tensor_conversion():
                tensor = torch.from_numpy(test_batch).float()
                tensor = tensor.permute(0, 3, 1, 2) / 255.0
                return tensor
            
            result = self.benchmark_function(tensor_conversion)
            result.name = f"tensor_conversion_batch_{batch_size}"
            results.append(result)
            
            # Benchmark GPU transfer (if available)
            if torch.cuda.is_available():
                def gpu_transfer():
                    tensor = torch.from_numpy(test_batch).float().cuda()
                    return tensor
                
                result = self.benchmark_function(gpu_transfer)
                result.name = f"gpu_transfer_batch_{batch_size}"
                results.append(result)
        
        return results
    
    def benchmark_memory_allocation(self) -> List[BenchmarkResult]:
        """Benchmark memory allocation patterns."""
        logger.info("Benchmarking memory allocation...")
        
        results = []
        
        # Test different array sizes
        array_sizes = [
            (1000, 1000),
            (2000, 2000),
            (3000, 3000),
            (4000, 4000)
        ]
        
        for width, height in array_sizes:
            logger.info(f"Testing array size: {width}x{height}")
            
            # Benchmark numpy array allocation
            def numpy_allocation():
                return np.random.rand(height, width, 3)
            
            result = self.benchmark_function(numpy_allocation)
            result.name = f"numpy_alloc_{width}x{height}"
            results.append(result)
            
            # Benchmark zeros allocation
            def zeros_allocation():
                return np.zeros((height, width, 3), dtype=np.float32)
            
            result = self.benchmark_function(zeros_allocation)
            result.name = f"zeros_alloc_{width}x{height}"
            results.append(result)
        
        return results
    
    def benchmark_video_processing(self) -> List[BenchmarkResult]:
        """Benchmark video processing operations."""
        logger.info("Benchmarking video processing...")
        
        results = []
        
        # Create test video frames
        frame_count = 100
        frame_size = (640, 480)
        test_frames = [
            np.random.randint(0, 255, (frame_size[1], frame_size[0], 3), dtype=np.uint8)
            for _ in range(frame_count)
        ]
        
        # Benchmark frame processing
        def process_frames():
            processed = []
            for frame in test_frames:
                # Simulate typical processing pipeline
                resized = cv2.resize(frame, (224, 224))
                normalized = resized.astype(np.float32) / 255.0
                processed.append(normalized)
            return processed
        
        result = self.benchmark_function(process_frames)
        result.name = f"video_processing_{frame_count}_frames"
        result.throughput = frame_count / result.execution_time
        results.append(result)
        
        return results
    
    def benchmark_io_operations(self) -> List[BenchmarkResult]:
        """Benchmark I/O operations."""
        logger.info("Benchmarking I/O operations...")
        
        results = []
        
        # Create test data
        test_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        test_data = {'detections': [{'x': 100, 'y': 100, 'w': 50, 'h': 50, 'score': 0.9}] * 100}
        
        # Benchmark image save/load
        def image_save_load():
            temp_path = '/tmp/test_image.jpg'
            cv2.imwrite(temp_path, test_image)
            loaded = cv2.imread(temp_path)
            os.unlink(temp_path)
            return loaded
        
        result = self.benchmark_function(image_save_load)
        result.name = "image_save_load"
        results.append(result)
        
        # Benchmark JSON save/load
        def json_save_load():
            temp_path = '/tmp/test_data.json'
            with open(temp_path, 'w') as f:
                json.dump(test_data, f, indent=2)
            with open(temp_path, 'r') as f:
                loaded = json.load(f)
            os.unlink(temp_path)
            return loaded
        
        result = self.benchmark_function(json_save_load)
        result.name = "json_save_load"
        results.append(result)
        
        return results
    
    def run_comprehensive_benchmark(self) -> Dict[str, List[BenchmarkResult]]:
        """Run comprehensive performance benchmark."""
        logger.info("Running comprehensive performance benchmark...")
        
        all_results = {}
        
        # Image processing benchmarks
        image_sizes = [(640, 480), (1280, 720), (1920, 1080), (3840, 2160)]
        all_results['image_processing'] = self.benchmark_image_processing(image_sizes)
        
        # Model inference benchmarks
        all_results['model_inference'] = self.benchmark_model_inference()
        
        # Memory allocation benchmarks
        all_results['memory_allocation'] = self.benchmark_memory_allocation()
        
        # Video processing benchmarks
        all_results['video_processing'] = self.benchmark_video_processing()
        
        # I/O operation benchmarks
        all_results['io_operations'] = self.benchmark_io_operations()
        
        return all_results
    
    def generate_report(self, results: Dict[str, List[BenchmarkResult]], output_path: str):
        """Generate benchmark report."""
        logger.info(f"Generating benchmark report: {output_path}")
        
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate summary report
        report = {
            'system_info': self._get_system_info(),
            'benchmark_results': {},
            'summary': {}
        }
        
        for category, benchmark_results in results.items():
            report['benchmark_results'][category] = []
            
            for result in benchmark_results:
                report['benchmark_results'][category].append({
                    'name': result.name,
                    'execution_time': result.execution_time,
                    'memory_usage': result.memory_usage,
                    'throughput': result.throughput,
                    'accuracy': result.accuracy,
                    'additional_metrics': result.additional_metrics
                })
            
            # Calculate category summary
            avg_time = np.mean([r.execution_time for r in benchmark_results])
            avg_memory = np.mean([r.memory_usage for r in benchmark_results])
            avg_throughput = np.mean([r.throughput for r in benchmark_results])
            
            report['summary'][category] = {
                'average_execution_time': avg_time,
                'average_memory_usage': avg_memory,
                'average_throughput': avg_throughput,
                'test_count': len(benchmark_results)
            }
        
        # Save JSON report
        with open(output_dir / 'benchmark_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate plots
        self._generate_plots(results, output_dir)
        
        # Generate text summary
        self._generate_text_summary(report, output_dir)
        
        logger.info(f"Benchmark report saved to {output_dir}")
    
    def _get_system_info(self) -> Dict:
        """Get system information."""
        import platform
        
        info = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'cuda_available': torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            info['cuda_device_count'] = torch.cuda.device_count()
            info['cuda_device_name'] = torch.cuda.get_device_name(0)
            info['cuda_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        return info
    
    def _generate_plots(self, results: Dict[str, List[BenchmarkResult]], output_dir: Path):
        """Generate performance plots."""
        try:
            import matplotlib.pyplot as plt
            
            # Execution time comparison
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Performance Benchmark Results', fontsize=16)
            
            # Plot 1: Execution time by category
            categories = list(results.keys())
            avg_times = [
                np.mean([r.execution_time for r in results[cat]]) 
                for cat in categories
            ]
            
            axes[0, 0].bar(categories, avg_times)
            axes[0, 0].set_title('Average Execution Time by Category')
            axes[0, 0].set_ylabel('Time (seconds)')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Plot 2: Memory usage by category
            avg_memory = [
                np.mean([r.memory_usage for r in results[cat]]) 
                for cat in categories
            ]
            
            axes[0, 1].bar(categories, avg_memory)
            axes[0, 1].set_title('Average Memory Usage by Category')
            axes[0, 1].set_ylabel('Memory (MB)')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Plot 3: Throughput by category
            avg_throughput = [
                np.mean([r.throughput for r in results[cat]]) 
                for cat in categories
            ]
            
            axes[1, 0].bar(categories, avg_throughput)
            axes[1, 0].set_title('Average Throughput by Category')
            axes[1, 0].set_ylabel('Operations/second')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Plot 4: Detailed view of image processing
            if 'image_processing' in results:
                img_results = results['image_processing']
                resize_results = [r for r in img_results if 'resize' in r.name]
                
                sizes = [r.name.split('_')[1] for r in resize_results]
                times = [r.execution_time for r in resize_results]
                
                axes[1, 1].plot(sizes, times, marker='o')
                axes[1, 1].set_title('Image Resize Performance')
                axes[1, 1].set_xlabel('Image Size')
                axes[1, 1].set_ylabel('Time (seconds)')
                axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'performance_plots.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Performance plots generated")
            
        except Exception as e:
            logger.error(f"Failed to generate plots: {e}")
    
    def _generate_text_summary(self, report: Dict, output_dir: Path):
        """Generate text summary."""
        with open(output_dir / 'benchmark_summary.txt', 'w') as f:
            f.write("Fire Prevention System - Performance Benchmark Report\n")
            f.write("=" * 60 + "\n\n")
            
            # System information
            f.write("System Information:\n")
            f.write("-" * 20 + "\n")
            for key, value in report['system_info'].items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # Summary by category
            f.write("Performance Summary:\n")
            f.write("-" * 20 + "\n")
            for category, summary in report['summary'].items():
                f.write(f"\n{category.upper()}:\n")
                f.write(f"  Average Execution Time: {summary['average_execution_time']:.4f}s\n")
                f.write(f"  Average Memory Usage: {summary['average_memory_usage']:.2f}MB\n")
                f.write(f"  Average Throughput: {summary['average_throughput']:.2f} ops/s\n")
                f.write(f"  Number of Tests: {summary['test_count']}\n")
            
            # Detailed results
            f.write("\n\nDetailed Results:\n")
            f.write("-" * 20 + "\n")
            for category, results in report['benchmark_results'].items():
                f.write(f"\n{category.upper()}:\n")
                for result in results:
                    f.write(f"  {result['name']}:\n")
                    f.write(f"    Time: {result['execution_time']:.4f}s\n")
                    f.write(f"    Memory: {result['memory_usage']:.2f}MB\n")
                    f.write(f"    Throughput: {result['throughput']:.2f} ops/s\n")
        
        logger.info("Text summary generated")


def main():
    parser = argparse.ArgumentParser(description='Fire Prevention System Performance Benchmark')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--output', default='benchmarks', help='Output directory for results')
    parser.add_argument('--category', 
                       choices=['image_processing', 'model_inference', 'memory_allocation', 
                               'video_processing', 'io_operations', 'all'],
                       default='all', help='Benchmark category to run')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    setup_logging(log_level)
    
    logger.info("Fire Prevention System - Performance Benchmark")
    logger.info("=" * 55)
    
    # Initialize benchmark
    benchmark = PerformanceBenchmark(args.config)
    
    # Run benchmarks
    if args.category == 'all':
        results = benchmark.run_comprehensive_benchmark()
    else:
        # Run specific category
        if args.category == 'image_processing':
            image_sizes = [(640, 480), (1280, 720), (1920, 1080)]
            results = {'image_processing': benchmark.benchmark_image_processing(image_sizes)}
        elif args.category == 'model_inference':
            results = {'model_inference': benchmark.benchmark_model_inference()}
        elif args.category == 'memory_allocation':
            results = {'memory_allocation': benchmark.benchmark_memory_allocation()}
        elif args.category == 'video_processing':
            results = {'video_processing': benchmark.benchmark_video_processing()}
        elif args.category == 'io_operations':
            results = {'io_operations': benchmark.benchmark_io_operations()}
    
    # Generate report
    benchmark.generate_report(results, args.output)
    
    # Print summary
    logger.info("\nBenchmark Summary:")
    logger.info("-" * 20)
    for category, benchmark_results in results.items():
        avg_time = np.mean([r.execution_time for r in benchmark_results])
        avg_memory = np.mean([r.memory_usage for r in benchmark_results])
        avg_throughput = np.mean([r.throughput for r in benchmark_results])
        
        logger.info(f"{category}:")
        logger.info(f"  Avg Time: {avg_time:.4f}s")
        logger.info(f"  Avg Memory: {avg_memory:.2f}MB")
        logger.info(f"  Avg Throughput: {avg_throughput:.2f} ops/s")
    
    logger.info(f"\nDetailed results saved to: {args.output}")
    logger.info("Benchmark completed successfully!")


if __name__ == "__main__":
    main()
