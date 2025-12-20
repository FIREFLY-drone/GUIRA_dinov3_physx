"""
System Monitoring and Health Check for Fire Prevention System
Provides real-time monitoring of system components and performance.
"""

import argparse
import time
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import threading
import queue
from datetime import datetime, timedelta
import psutil
import torch
import GPUtil

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from utils import setup_logging, load_config
from loguru import logger


class SystemMonitor:
    """Monitors system health and performance."""
    
    def __init__(self, config_path: str = 'config.yaml', update_interval: float = 5.0):
        self.config = load_config(config_path)
        self.update_interval = update_interval
        self.monitoring = False
        self.metrics_queue = queue.Queue(maxsize=1000)
        self.alert_thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'gpu_usage': 90.0,
            'gpu_memory': 90.0,
            'disk_usage': 90.0,
            'temperature': 80.0
        }
        
    def get_system_metrics(self) -> Dict:
        """Get current system metrics."""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'cpu': self._get_cpu_metrics(),
            'memory': self._get_memory_metrics(),
            'disk': self._get_disk_metrics(),
            'gpu': self._get_gpu_metrics(),
            'network': self._get_network_metrics(),
            'processes': self._get_process_metrics()
        }
        
        return metrics
    
    def _get_cpu_metrics(self) -> Dict:
        """Get CPU metrics."""
        return {
            'usage_percent': psutil.cpu_percent(interval=1),
            'count': psutil.cpu_count(),
            'frequency': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
        }
    
    def _get_memory_metrics(self) -> Dict:
        """Get memory metrics."""
        virtual = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            'virtual': {
                'total_gb': virtual.total / (1024**3),
                'available_gb': virtual.available / (1024**3),
                'used_gb': virtual.used / (1024**3),
                'usage_percent': virtual.percent
            },
            'swap': {
                'total_gb': swap.total / (1024**3),
                'used_gb': swap.used / (1024**3),
                'usage_percent': swap.percent
            }
        }
    
    def _get_disk_metrics(self) -> Dict:
        """Get disk metrics."""
        disk_usage = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        metrics = {
            'usage': {
                'total_gb': disk_usage.total / (1024**3),
                'used_gb': disk_usage.used / (1024**3),
                'free_gb': disk_usage.free / (1024**3),
                'usage_percent': (disk_usage.used / disk_usage.total) * 100
            }
        }
        
        if disk_io:
            metrics['io'] = {
                'read_bytes': disk_io.read_bytes,
                'write_bytes': disk_io.write_bytes,
                'read_count': disk_io.read_count,
                'write_count': disk_io.write_count
            }
        
        return metrics
    
    def _get_gpu_metrics(self) -> Dict:
        """Get GPU metrics."""
        if not torch.cuda.is_available():
            return {'available': False}
        
        try:
            gpus = GPUtil.getGPUs()
            gpu_metrics = {
                'available': True,
                'count': torch.cuda.device_count(),
                'devices': []
            }
            
            for i, gpu in enumerate(gpus):
                device_metrics = {
                    'id': gpu.id,
                    'name': gpu.name,
                    'memory_total_mb': gpu.memoryTotal,
                    'memory_used_mb': gpu.memoryUsed,
                    'memory_free_mb': gpu.memoryFree,
                    'memory_usage_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    'gpu_usage_percent': gpu.load * 100,
                    'temperature': gpu.temperature
                }
                gpu_metrics['devices'].append(device_metrics)
            
            return gpu_metrics
            
        except Exception as e:
            logger.warning(f"Failed to get GPU metrics: {e}")
            return {
                'available': True,
                'error': str(e),
                'count': torch.cuda.device_count()
            }
    
    def _get_network_metrics(self) -> Dict:
        """Get network metrics."""
        net_io = psutil.net_io_counters()
        
        if net_io:
            return {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv,
                'errors_in': net_io.errin,
                'errors_out': net_io.errout,
                'drops_in': net_io.dropin,
                'drops_out': net_io.dropout
            }
        else:
            return {'available': False}
    
    def _get_process_metrics(self) -> Dict:
        """Get process-specific metrics."""
        current_process = psutil.Process()
        
        try:
            return {
                'pid': current_process.pid,
                'memory_mb': current_process.memory_info().rss / (1024 * 1024),
                'cpu_percent': current_process.cpu_percent(),
                'num_threads': current_process.num_threads(),
                'num_fds': current_process.num_fds() if hasattr(current_process, 'num_fds') else None,
                'create_time': current_process.create_time(),
                'status': current_process.status()
            }
        except Exception as e:
            logger.warning(f"Failed to get process metrics: {e}")
            return {'error': str(e)}
    
    def check_model_status(self) -> Dict:
        """Check status of trained models."""
        models_dir = Path('models')
        model_status = {
            'models_directory_exists': models_dir.exists(),
            'models': {}
        }
        
        if models_dir.exists():
            model_files = list(models_dir.glob('*.pt'))
            for model_file in model_files:
                stat = model_file.stat()
                model_status['models'][model_file.name] = {
                    'size_mb': stat.st_size / (1024 * 1024),
                    'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'accessible': model_file.is_file()
                }
        
        return model_status
    
    def check_data_status(self) -> Dict:
        """Check status of data directories."""
        data_dir = Path('data')
        data_status = {
            'data_directory_exists': data_dir.exists(),
            'subdirectories': {}
        }
        
        if data_dir.exists():
            required_dirs = ['fire', 'smoke', 'fauna', 'vegetation', 'pose', 'dem', 'spread']
            
            for dir_name in required_dirs:
                subdir = data_dir / dir_name
                if subdir.exists():
                    file_count = len(list(subdir.rglob('*')))
                    total_size = sum(f.stat().st_size for f in subdir.rglob('*') if f.is_file())
                    
                    data_status['subdirectories'][dir_name] = {
                        'exists': True,
                        'file_count': file_count,
                        'total_size_mb': total_size / (1024 * 1024)
                    }
                else:
                    data_status['subdirectories'][dir_name] = {'exists': False}
        
        return data_status
    
    def check_dependencies(self) -> Dict:
        """Check status of dependencies."""
        dependencies = {
            'python_packages': {},
            'system_libraries': {}
        }
        
        # Check key Python packages
        packages_to_check = [
            'torch', 'torchvision', 'tensorflow', 'opencv-python', 
            'ultralytics', 'transformers', 'numpy', 'pandas',
            'rasterio', 'gdal', 'geopandas', 'loguru'
        ]
        
        for package in packages_to_check:
            try:
                module = __import__(package.replace('-', '_'))
                version = getattr(module, '__version__', 'unknown')
                dependencies['python_packages'][package] = {
                    'installed': True,
                    'version': version
                }
            except ImportError:
                dependencies['python_packages'][package] = {
                    'installed': False,
                    'version': None
                }
        
        return dependencies
    
    def detect_alerts(self, metrics: Dict) -> List[Dict]:
        """Detect system alerts based on thresholds."""
        alerts = []
        
        # CPU usage alert
        cpu_usage = metrics['cpu']['usage_percent']
        if cpu_usage > self.alert_thresholds['cpu_usage']:
            alerts.append({
                'type': 'high_cpu_usage',
                'severity': 'warning',
                'message': f'High CPU usage: {cpu_usage:.1f}%',
                'value': cpu_usage,
                'threshold': self.alert_thresholds['cpu_usage']
            })
        
        # Memory usage alert
        memory_usage = metrics['memory']['virtual']['usage_percent']
        if memory_usage > self.alert_thresholds['memory_usage']:
            alerts.append({
                'type': 'high_memory_usage',
                'severity': 'warning',
                'message': f'High memory usage: {memory_usage:.1f}%',
                'value': memory_usage,
                'threshold': self.alert_thresholds['memory_usage']
            })
        
        # GPU alerts
        if metrics['gpu']['available'] and 'devices' in metrics['gpu']:
            for gpu in metrics['gpu']['devices']:
                gpu_usage = gpu['gpu_usage_percent']
                gpu_memory = gpu['memory_usage_percent']
                
                if gpu_usage > self.alert_thresholds['gpu_usage']:
                    alerts.append({
                        'type': 'high_gpu_usage',
                        'severity': 'warning',
                        'message': f'High GPU usage on {gpu["name"]}: {gpu_usage:.1f}%',
                        'value': gpu_usage,
                        'threshold': self.alert_thresholds['gpu_usage']
                    })
                
                if gpu_memory > self.alert_thresholds['gpu_memory']:
                    alerts.append({
                        'type': 'high_gpu_memory',
                        'severity': 'warning',
                        'message': f'High GPU memory on {gpu["name"]}: {gpu_memory:.1f}%',
                        'value': gpu_memory,
                        'threshold': self.alert_thresholds['gpu_memory']
                    })
        
        # Disk usage alert
        disk_usage = metrics['disk']['usage']['usage_percent']
        if disk_usage > self.alert_thresholds['disk_usage']:
            alerts.append({
                'type': 'high_disk_usage',
                'severity': 'critical',
                'message': f'High disk usage: {disk_usage:.1f}%',
                'value': disk_usage,
                'threshold': self.alert_thresholds['disk_usage']
            })
        
        return alerts
    
    def start_monitoring(self, output_file: Optional[str] = None):
        """Start continuous monitoring."""
        self.monitoring = True
        logger.info("Starting system monitoring...")
        
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            while self.monitoring:
                # Get metrics
                metrics = self.get_system_metrics()
                
                # Check for alerts
                alerts = self.detect_alerts(metrics)
                
                # Log alerts
                for alert in alerts:
                    if alert['severity'] == 'critical':
                        logger.error(f"🚨 {alert['message']}")
                    else:
                        logger.warning(f"⚠️  {alert['message']}")
                
                # Store metrics
                monitor_data = {
                    'metrics': metrics,
                    'alerts': alerts,
                    'health_status': 'healthy' if not alerts else 'degraded'
                }
                
                # Add to queue
                try:
                    self.metrics_queue.put_nowait(monitor_data)
                except queue.Full:
                    # Remove oldest item
                    try:
                        self.metrics_queue.get_nowait()
                        self.metrics_queue.put_nowait(monitor_data)
                    except queue.Empty:
                        pass
                
                # Save to file if specified
                if output_file:
                    with open(output_path, 'a') as f:
                        f.write(json.dumps(monitor_data) + '\n')
                
                # Wait for next update
                time.sleep(self.update_interval)
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
        finally:
            self.monitoring = False
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.monitoring = False
        logger.info("Stopping system monitoring...")
    
    def get_health_summary(self) -> Dict:
        """Get comprehensive health summary."""
        logger.info("Generating system health summary...")
        
        # Get current metrics
        metrics = self.get_system_metrics()
        alerts = self.detect_alerts(metrics)
        
        # Check models
        model_status = self.check_model_status()
        
        # Check data
        data_status = self.check_data_status()
        
        # Check dependencies
        dependencies = self.check_dependencies()
        
        # Overall health assessment
        health_score = self._calculate_health_score(metrics, model_status, data_status, dependencies)
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'overall_health': {
                'status': 'healthy' if health_score > 0.8 else 'degraded' if health_score > 0.6 else 'critical',
                'score': health_score,
                'alert_count': len(alerts)
            },
            'system_metrics': metrics,
            'alerts': alerts,
            'model_status': model_status,
            'data_status': data_status,
            'dependencies': dependencies,
            'recommendations': self._generate_recommendations(metrics, alerts, model_status, data_status)
        }
        
        return summary
    
    def _calculate_health_score(self, metrics: Dict, model_status: Dict, 
                               data_status: Dict, dependencies: Dict) -> float:
        """Calculate overall system health score (0-1)."""
        score = 1.0
        
        # Deduct for high resource usage
        cpu_usage = metrics['cpu']['usage_percent']
        if cpu_usage > 80:
            score -= 0.1
        elif cpu_usage > 90:
            score -= 0.2
        
        memory_usage = metrics['memory']['virtual']['usage_percent']
        if memory_usage > 85:
            score -= 0.1
        elif memory_usage > 95:
            score -= 0.2
        
        # Deduct for missing models
        if not model_status['models_directory_exists']:
            score -= 0.3
        elif len(model_status['models']) == 0:
            score -= 0.2
        
        # Deduct for missing data
        if not data_status['data_directory_exists']:
            score -= 0.2
        else:
            missing_dirs = sum(1 for status in data_status['subdirectories'].values() 
                             if not status['exists'])
            score -= missing_dirs * 0.05
        
        # Deduct for missing dependencies
        missing_packages = sum(1 for status in dependencies['python_packages'].values() 
                             if not status['installed'])
        score -= missing_packages * 0.02
        
        return max(0.0, score)
    
    def _generate_recommendations(self, metrics: Dict, alerts: List[Dict], 
                                 model_status: Dict, data_status: Dict) -> List[str]:
        """Generate system recommendations."""
        recommendations = []
        
        # Resource recommendations
        if metrics['cpu']['usage_percent'] > 80:
            recommendations.append("Consider reducing batch sizes or using fewer parallel workers")
        
        if metrics['memory']['virtual']['usage_percent'] > 85:
            recommendations.append("Consider reducing cache sizes or clearing unused model weights")
        
        # Model recommendations
        if not model_status['models_directory_exists'] or len(model_status['models']) == 0:
            recommendations.append("Train models using: python cli.py train --model all")
        
        # Data recommendations
        if not data_status['data_directory_exists']:
            recommendations.append("Download datasets using: python cli.py download --dataset all")
        
        missing_dirs = [name for name, status in data_status.get('subdirectories', {}).items() 
                       if not status['exists']]
        if missing_dirs:
            recommendations.append(f"Missing data directories: {', '.join(missing_dirs)}")
        
        # GPU recommendations
        if metrics['gpu']['available'] and 'devices' in metrics['gpu']:
            for gpu in metrics['gpu']['devices']:
                if gpu['memory_usage_percent'] > 90:
                    recommendations.append(f"GPU {gpu['name']} memory is high - consider using mixed precision")
        
        return recommendations


def main():
    parser = argparse.ArgumentParser(description='Fire Prevention System Monitor')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--mode', choices=['monitor', 'health', 'continuous'],
                       default='health', help='Monitoring mode')
    parser.add_argument('--interval', type=float, default=5.0, help='Update interval for continuous monitoring')
    parser.add_argument('--output', help='Output file for continuous monitoring')
    parser.add_argument('--duration', type=int, help='Duration in seconds for continuous monitoring')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging('INFO')
    
    logger.info("Fire Prevention System - System Monitor")
    logger.info("=" * 50)
    
    # Initialize monitor
    monitor = SystemMonitor(args.config, args.interval)
    
    if args.mode == 'health':
        # Single health check
        summary = monitor.get_health_summary()
        
        # Print summary
        logger.info(f"Overall Health: {summary['overall_health']['status'].upper()}")
        logger.info(f"Health Score: {summary['overall_health']['score']:.2f}")
        logger.info(f"Active Alerts: {summary['overall_health']['alert_count']}")
        
        # Print alerts
        if summary['alerts']:
            logger.info("\nActive Alerts:")
            for alert in summary['alerts']:
                logger.warning(f"  - {alert['message']}")
        
        # Print recommendations
        if summary['recommendations']:
            logger.info("\nRecommendations:")
            for rec in summary['recommendations']:
                logger.info(f"  - {rec}")
        
        # Save detailed report
        output_path = Path('system_health_report.json')
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"\nDetailed report saved to: {output_path}")
    
    elif args.mode == 'monitor':
        # Single monitoring snapshot
        metrics = monitor.get_system_metrics()
        alerts = monitor.detect_alerts(metrics)
        
        logger.info("Current System Metrics:")
        logger.info(f"  CPU Usage: {metrics['cpu']['usage_percent']:.1f}%")
        logger.info(f"  Memory Usage: {metrics['memory']['virtual']['usage_percent']:.1f}%")
        logger.info(f"  Disk Usage: {metrics['disk']['usage']['usage_percent']:.1f}%")
        
        if metrics['gpu']['available'] and 'devices' in metrics['gpu']:
            for gpu in metrics['gpu']['devices']:
                logger.info(f"  GPU {gpu['name']}: {gpu['gpu_usage_percent']:.1f}% usage, "
                           f"{gpu['memory_usage_percent']:.1f}% memory")
        
        if alerts:
            logger.info("\nActive Alerts:")
            for alert in alerts:
                logger.warning(f"  - {alert['message']}")
    
    elif args.mode == 'continuous':
        # Continuous monitoring
        logger.info(f"Starting continuous monitoring (interval: {args.interval}s)")
        if args.output:
            logger.info(f"Logging to: {args.output}")
        
        if args.duration:
            # Start monitoring in a thread
            monitor_thread = threading.Thread(
                target=monitor.start_monitoring, 
                args=(args.output,)
            )
            monitor_thread.start()
            
            # Wait for specified duration
            time.sleep(args.duration)
            
            # Stop monitoring
            monitor.stop_monitoring()
            monitor_thread.join()
            
            logger.info(f"Monitoring completed after {args.duration} seconds")
        else:
            # Run until interrupted
            try:
                monitor.start_monitoring(args.output)
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
    
    logger.info("System monitoring completed")


if __name__ == "__main__":
    main()
