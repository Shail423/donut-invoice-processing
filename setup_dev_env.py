#!/usr/bin/env python
"""
Setup script to quickly configure development environment for the Donut project.
Run this script after cloning the repository to set up your environment.
"""
import os
import sys
import subprocess
import platform

def run_command(command):
    """Run a shell command and print output."""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    print(result.stdout)
    return True

def setup_environment():
    """Set up the development environment."""
    print("Setting up Donut development environment...")
    
    # Create virtual environment
    if platform.system() == "Windows":
        venv_command = "python -m venv donut-env"
        activate_command = ".\\donut-env\\Scripts\\activate"
    else:
        venv_command = "python3 -m venv donut-env"
        activate_command = "source ./donut-env/bin/activate"
    
    if not run_command(venv_command):
        print("Failed to create virtual environment")
        return False
    
    print(f"Virtual environment created. Activate it with: {activate_command}")
    
    # Install dependencies
    if platform.system() == "Windows":
        pip_command = ".\\donut-env\\Scripts\\pip install -r requirements.txt"
    else:
        pip_command = "./donut-env/bin/pip install -r requirements.txt"
    
    if not run_command(pip_command):
        print("Failed to install dependencies")
        return False
    
    # Create necessary directories
    directories = ["data/images", "data/output_json", "logs"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    print("\nEnvironment setup complete!")
    print("\nNext steps:")
    print(f"1. Activate the virtual environment: {activate_command}")
    print("2. Download model files to donut-invoice-model/ directory")
    print("3. Place invoice images in data/images/ directory")
    print("4. Run 'python app.py' to start the application")
    
    return True

if __name__ == "__main__":
    setup_environment()