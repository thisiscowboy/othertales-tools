#!/usr/bin/env python3
"""
Dependency checker script to help identify and resolve package incompatibilities
"""
import sys
import subprocess
import pkg_resources

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

if __name__ == "__main__":
    if check_dependencies():
        sys.exit(0)
    else:
        sys.exit(1)
