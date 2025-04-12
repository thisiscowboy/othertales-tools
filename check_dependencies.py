#!/usr/bin/env python3
"""
Dependency checker script to help identify and resolve package incompatibilities
"""
import sys
import subprocess
import pkg_resources
import importlib
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

def check_dependencies():
    """Check if all dependencies can be installed without conflicts"""
    try:
        # Read requirements file
        with open("requirements.txt", "r") as f:
            requirements = f.read().splitlines()
        
        # Filter out comments and empty lines
        requirements = [line for line in requirements if line and not line.startswith("#")]
        
        # Check if dependencies can be resolved
        pkg_resources.working_set.resolve([pkg_resources.Requirement.parse(r) for r in requirements])
        print("All dependencies can be resolved successfully!")
        return True
    except pkg_resources.DistributionNotFound as e:
        print(f"Missing dependency: {e}")
        return False
    except pkg_resources.VersionConflict as e:
        print(f"Version conflict: {e}")
        return False

def check_vector_dependencies():
    """Check if vector dependencies are installed and properly working"""
    print("\n=== Vector Search Dependencies Check ===\n")
    
    # Check numpy
    try:
        import numpy
        print(f"✓ numpy is installed (version: {numpy.__version__})")
    except ImportError:
        print("× numpy is NOT installed")
        return False
    
    # Check torch
    try:
        import torch
        print(f"✓ torch is installed (version: {torch.__version__})")
    except ImportError:
        print("× torch is NOT installed")
        return False
    
    # Check sentence-transformers
    try:
        import sentence_transformers
        print(f"✓ sentence-transformers is installed (version: {sentence_transformers.__version__})")
    except ImportError:
        print("× sentence-transformers is NOT installed")
        return False
    
    # Check if we can initialize a model
    try:
        print("Testing SentenceTransformer initialization...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print("✓ Model initialized successfully")
        
        # Test encoding
        test_text = "This is a test sentence for vector embedding."
        print("Testing text encoding...")
        embedding = model.encode(test_text)
        print(f"✓ Encoding successful (embedding shape: {embedding.shape})")
        
        return True
    except Exception as e:
        print(f"× Error initializing or using SentenceTransformer: {e}")
        return False

def check_installed_packages():
    """Check installed packages against requirements.txt"""
    # Get installed packages
    installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    
    # Read requirements
    with open("requirements.txt", "r") as f:
        requirements = f.read().splitlines()
    
    # Filter out comments and empty lines
    requirements = [line for line in requirements if line and not line.startswith("#")]
    
    print("\n=== Package Installation Status ===\n")
    
    for req in requirements:
        # Split off version specifiers
        pkg_name = req.split('==')[0].split('>=')[0].split('>')[0].strip().lower()
        
        if pkg_name in installed_packages:
            print(f"✓ {pkg_name} (installed: {installed_packages[pkg_name]})")
        else:
            print(f"× {pkg_name} (NOT installed)")

if __name__ == "__main__":
    print(f"\nPython version: {sys.version}")
    print(f"Running from: {os.getcwd()}\n")
    
    basic_deps_ok = check_dependencies()
    check_installed_packages()
    vector_deps_ok = check_vector_dependencies()
    
    print("\n=== Final Status ===")
    if basic_deps_ok:
        print("✓ Basic dependencies check passed")
    else:
        print("× Basic dependencies check failed")
    
    if vector_deps_ok:
        print("✓ Vector search dependencies check passed")
    else:
        print("× Vector search dependencies check failed")
        print("\nTo fix vector search dependencies, run:")
        print("pip install numpy==2.2.4 torch==2.4.1 sentence-transformers==2.2.2")
    
    if not basic_deps_ok or not vector_deps_ok:
        sys.exit(1)
    
    print("\nAll checks passed successfully!")
    sys.exit(0)
