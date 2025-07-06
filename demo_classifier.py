#!/usr/bin/env python3
"""
Demo script for the Animal Classifier

This script demonstrates how to use the animal_classifier.py script
and provides interactive functionality for testing.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_classification_with_file(image_path, model_type, model_file, top_k=3):
    """Run the animal classifier on an image with specific model file"""
    cmd = [
        sys.executable, "animal_classifier.py",
        "--image", image_path,
        "--model", model_type,
        "--model-file", model_file,
        "--top-k", str(top_k)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
    except Exception as e:
        print(f"Error running classifier: {e}")

def list_image_files(directory="."):
    """List common image files in the directory"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    image_files = []
    
    try:
        for file in os.listdir(directory):
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(file)
    except Exception as e:
        print(f"Error listing files: {e}")
    
    return sorted(image_files)

def check_requirements():
    """Check if required files and dependencies are available"""
    print("Checking requirements...")
    
    # Check if main script exists
    if not os.path.exists("animal_classifier.py"):
        print("ERROR: animal_classifier.py not found!")
        return False
    
    # Check if model files exist
    model_files = [f for f in os.listdir('.') if f.endswith('.pth')]
    if not model_files:
        print("ERROR: No model files (.pth) found!")
        return False
    
    print(f"Found {len(model_files)} model files")
    
    # Check Python dependencies
    try:
        import torch
        import torchvision
        from PIL import Image
        print("All required Python packages are available")
    except ImportError as e:
        print(f"ERROR: Missing Python package: {e}")
        return False
    
    return True

def interactive_demo():
    """Run an interactive demo"""
    print("Animal Classifier - Interactive Demo")
    print("=" * 40)
    
    if not check_requirements():
        print("\nPlease install missing requirements and ensure model files are present.")
        return
    
    # List available images
    image_files = list_image_files()
    if image_files:
        print(f"\nFound {len(image_files)} image files in current directory:")
        for i, img in enumerate(image_files, 1):
            print(f"  {i}. {img}")
    
    # List available models
    model_files = [f for f in os.listdir('.') if f.endswith('.pth')]
    available_models = set()
    for model_file in model_files:
        if model_file.startswith('resnet50_'):
            available_models.add('resnet50')
        elif model_file.startswith('vgg16_'):
            available_models.add('vgg16')
        elif model_file.startswith('inceptionv3_'):
            available_models.add('inceptionv3')
    
    print(f"\nAvailable models: {', '.join(sorted(available_models))}")
    
    while True:
        print("\n" + "-" * 40)
        image_path = input("Enter image path (or 'quit' to exit): ").strip()
        
        if image_path.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
            
        if not os.path.exists(image_path):
            print(f"Error: File '{image_path}' not found!")
            continue
        
        # Show available model files
        model_files = [f for f in os.listdir('.') if f.endswith('.pth')]
        print(f"\nAvailable model files: {', '.join(sorted(model_files))}")
        
        model_file = input("Enter model filename (e.g., resnet50_fold1.pth): ").strip()
        
        # Add .pth extension if not provided
        if not model_file.endswith('.pth'):
            model_file += '.pth'
            
        if not os.path.exists(model_file):
            print(f"Error: Model file '{model_file}' not found!")
            continue
        
        # Extract model type from filename for transforms
        if model_file.startswith('resnet50_'):
            model_type = 'resnet50'
        elif model_file.startswith('vgg16_'):
            model_type = 'vgg16'
        elif model_file.startswith('inceptionv3_'):
            model_type = 'inceptionv3'
        else:
            print(f"Error: Unsupported model file format: {model_file}")
            continue
        
        print(f"\nClassifying '{image_path}' with {model_file}...")
        run_classification_with_file(image_path, model_type, model_file)


def quick_test():
    """Run a quick test if image files are available"""
    print("Quick Test Mode")
    print("-" * 20)
    
    # Find first available image and model
    image_files = list_image_files()
    if not image_files:
        print("No image files found for testing.")
        return
    
    model_files = [f for f in os.listdir('.') if f.endswith('.pth')]
    if not model_files:
        print("No model files found for testing.")
        return
    
    # Use first available image and ResNet50 if available
    test_image = image_files[0]
    test_model = 'resnet50'
    
    # Check if ResNet50 is available, otherwise use first available model
    resnet_files = [f for f in model_files if f.startswith('resnet50_')]
    if not resnet_files:
        if any(f.startswith('vgg16_') for f in model_files):
            test_model = 'vgg16'
        elif any(f.startswith('inceptionv3_') for f in model_files):
            test_model = 'inceptionv3'
        else:
            print("No compatible model files found.")
            return
    
    print(f"Testing with: {test_image} using {test_model}")
    model_file = f"{test_model}_fold1.pth"
    run_classification_with_file(test_image, test_model, model_file)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        quick_test()
    else:
        interactive_demo() 