#!/usr/bin/env python3
"""
System Validation Script for Fire Prevention System
Validates all components and dependencies are working correctly.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def check_directories():
    """Check if all required directories exist."""
    print("\n📁 Checking directory structure...")
    required_dirs = [
        "src", "src/fire", "src/smoke", "src/fauna", "src/vegetation", 
        "src/geospatial", "src/spread", "utils", "config", "data", 
        "models", "logs", "outputs"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"   ✅ {dir_path}")
        else:
            print(f"   ❌ {dir_path} - Missing")
            all_exist = False
    
    return all_exist

def check_core_files():
    """Check if all core files exist."""
    print("\n📄 Checking core files...")
    core_files = [
        "config.yaml",
        "requirements.txt",
        "cli.py",
        "run_pipeline.py",
        "download_datasets.py",
        "test_system.py",
        "benchmark.py",
        "monitor.py",
        "src/fire/fire_detection.py",
        "src/smoke/smoke_detection.py",
        "src/fauna/fauna_detection.py",
        "src/vegetation/vegetation_health.py",
        "src/geospatial/geospatial_projection.py",
        "src/spread/fire_spread_simulation.py",
        "config/intrinsics.json"
    ]
    
    all_exist = True
    for file_path in core_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - Missing")
            all_exist = False
    
    return all_exist

def check_import_capabilities():
    """Check if basic Python imports work."""
    print("\n🔗 Checking basic import capabilities...")
    basic_imports = [
        ("os", "Operating System interface"),
        ("sys", "System-specific parameters"),
        ("json", "JSON encoder/decoder"),
        ("yaml", "YAML parser", "pyyaml"),
        ("cv2", "OpenCV", "opencv-python"),
        ("PIL", "Python Imaging Library", "pillow"),
        ("numpy", "NumPy", "numpy"),
        ("matplotlib", "Matplotlib", "matplotlib")
    ]
    
    import_success = True
    for item in basic_imports:
        module_name = item[0]
        description = item[1]
        package_name = item[2] if len(item) > 2 else module_name
        
        try:
            __import__(module_name)
            print(f"   ✅ {module_name} ({description})")
        except ImportError:
            print(f"   ❌ {module_name} ({description}) - Install with: pip install {package_name}")
            import_success = False
    
    return import_success

def check_optional_dependencies():
    """Check optional deep learning dependencies."""
    print("\n🤖 Checking optional ML dependencies...")
    ml_imports = [
        ("torch", "PyTorch", "torch"),
        ("torchvision", "TorchVision", "torchvision"),
        ("sklearn", "Scikit-learn", "scikit-learn"),
        ("tqdm", "Progress bars", "tqdm")
    ]
    
    ml_success = 0
    for item in ml_imports:
        module_name = item[0]
        description = item[1]
        package_name = item[2]
        
        try:
            __import__(module_name)
            print(f"   ✅ {module_name} ({description})")
            ml_success += 1
        except ImportError:
            print(f"   ⚠️  {module_name} ({description}) - Optional: pip install {package_name}")
    
    return ml_success

def create_sample_data():
    """Create sample data files for testing."""
    print("\n📊 Creating sample data...")
    
    # Create sample config files
    sample_paths = [
        "data/fire/sample.txt",
        "data/smoke/sample.txt", 
        "data/fauna/sample.txt",
        "data/vegetation/sample.txt",
        "data/spread/sample.txt",
        "models/README.md",
        "outputs/README.md",
        "logs/README.md"
    ]
    
    for path in sample_paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            with open(path, 'w') as f:
                if path.endswith('.txt'):
                    f.write(f"Sample data placeholder for {os.path.dirname(path)}\n")
                else:
                    f.write(f"# {os.path.basename(os.path.dirname(path)).title()} Directory\n\n")
                    f.write(f"This directory contains {os.path.basename(os.path.dirname(path))} files.\n")
            print(f"   ✅ Created {path}")
        else:
            print(f"   ✅ {path} already exists")

def main():
    """Run complete system validation."""
    print("🔥 Fire Prevention System - Validation Report")
    print("=" * 50)
    
    # Run all checks
    checks = [
        ("Python Version", check_python_version),
        ("Directory Structure", check_directories),
        ("Core Files", check_core_files),
        ("Basic Imports", check_import_capabilities),
    ]
    
    results = {}
    for check_name, check_func in checks:
        results[check_name] = check_func()
    
    # Optional ML dependencies
    ml_count = check_optional_dependencies()
    results["ML Dependencies"] = ml_count >= 2  # At least 2 out of 4
    
    # Create sample data
    create_sample_data()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for check_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name:<20}: {status}")
    
    print(f"\nOverall Status: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 System validation completed successfully!")
        print("   Ready to proceed with training and inference.")
        return True
    else:
        print(f"\n⚠️  System validation incomplete ({passed}/{total})")
        print("   Please resolve the issues above before proceeding.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
