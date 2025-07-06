#!/usr/bin/env python3
"""
Animal Classification Inference Script

This script loads trained models (ResNet50, VGG16, or InceptionV3) and classifies animal images.
The models can predict from 90 different animal classes.

Usage:
    python animal_classifier.py --image path/to/image.jpg --model resnet50
    python animal_classifier.py --image path/to/image.jpg --model vgg16
    python animal_classifier.py --image path/to/image.jpg --model inceptionv3
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import argparse
import sys
import os

# Animal class names (sorted alphabetically as used in training)
ANIMAL_CLASSES = [
    'antelope', 'badger', 'bat', 'bear', 'bee', 'beetle', 'bison', 'boar', 
    'butterfly', 'cat', 'caterpillar', 'chimpanzee', 'cockroach', 'cow', 
    'coyote', 'crab', 'crow', 'deer', 'dog', 'dolphin', 'donkey', 'dragonfly', 
    'duck', 'eagle', 'elephant', 'flamingo', 'fly', 'fox', 'goat', 'goldfish', 
    'goose', 'gorilla', 'grasshopper', 'hamster', 'hare', 'hedgehog', 
    'hippopotamus', 'hornbill', 'horse', 'hummingbird', 'hyena', 'jellyfish', 
    'kangaroo', 'koala', 'ladybugs', 'leopard', 'lion', 'lizard', 'lobster', 
    'mosquito', 'moth', 'mouse', 'octopus', 'okapi', 'orangutan', 'otter', 
    'owl', 'ox', 'oyster', 'panda', 'parrot', 'pelecaniformes', 'penguin', 
    'pig', 'pigeon', 'porcupine', 'possum', 'raccoon', 'rat', 'reindeer', 
    'rhinoceros', 'sandpiper', 'seahorse', 'seal', 'shark', 'sheep', 'snake', 
    'sparrow', 'squid', 'squirrel', 'starfish', 'swan', 'tiger', 'turkey', 
    'turtle', 'whale', 'wolf', 'wombat', 'woodpecker', 'zebra'
]

NUM_CLASSES = len(ANIMAL_CLASSES)

# Image preprocessing transforms
def get_transforms(model_name):
    """Get the appropriate transforms for each model"""
    if model_name.lower() == 'inceptionv3':
        # InceptionV3 uses 299x299 input size
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # ResNet50 and VGG16 use 224x224 input size
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    return transform

def load_model(model_name, model_path, device):
    """Load the specified pre-trained model"""
    model_name = model_name.lower()
    
    if model_name == 'resnet50':
        model = models.resnet50(weights=None)  # Don't load pretrained weights
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    elif model_name == 'vgg16':
        model = models.vgg16(weights=None)  # Don't load pretrained weights
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, NUM_CLASSES)  # type: ignore
    elif model_name == 'inceptionv3':
        model = models.inception_v3(weights=None)  # Don't load pretrained weights
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        model.aux_logits = False  # Disable auxiliary classifier for inference
    else:
        raise ValueError(f"Unsupported model: {model_name}. Choose from: resnet50, vgg16, inceptionv3")
    
    # Load the trained weights
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Successfully loaded model: {model_name}")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return None
    
    model.to(device)
    model.eval()
    return model

def predict_image(model, image_path, transform, device, top_k=3):
    """Predict the class of an image"""
    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, top_k)
        
        predictions = []
        for i in range(top_k):
            class_idx = int(top_indices[0][i].item())
            confidence = top_probs[0][i].item()
            class_name = ANIMAL_CLASSES[class_idx]
            predictions.append((class_name, confidence))
        
        return predictions
    
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Classify animal images using trained models')
    parser.add_argument('--image', required=True, help='Path to the input image')
    parser.add_argument('--model', required=True, choices=['resnet50', 'vgg16', 'inceptionv3'],
                       help='Model to use for classification')
    parser.add_argument('--fold', default=1, type=int, choices=[1, 2, 3, 4, 5],
                       help='Which fold model to use (default: 1)')
    parser.add_argument('--model-file', help='Direct path to model file (overrides --model and --fold)')
    parser.add_argument('--top-k', default=3, type=int,
                       help='Number of top predictions to show (default: 3)')
    
    args = parser.parse_args()
    
    # Check if image file exists
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found!")
        sys.exit(1)
    
    # Determine model path
    if hasattr(args, 'model_file') and args.model_file:
        model_path = args.model_file
    else:
        model_path = f"{args.model}_fold{args.fold}.pth"
    
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print(f"Available model files: {[f for f in os.listdir('.') if f.endswith('.pth')]}")
        sys.exit(1)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model, model_path, device)
    if model is None:
        sys.exit(1)
    
    # Get transforms
    transform = get_transforms(args.model)
    
    # Make prediction
    print(f"\nClassifying image: {args.image}")
    if hasattr(args, 'model_file') and args.model_file:
        print(f"Using model file: {model_path}")
    else:
        print(f"Using model: {args.model} (fold {args.fold})")
    print("-" * 50)
    
    predictions = predict_image(model, args.image, transform, device, args.top_k)
    
    if predictions:
        print(f"Top {args.top_k} predictions:")
        for i, (class_name, confidence) in enumerate(predictions, 1):
            print(f"{i}. {class_name}: {confidence:.2%}")
        
        # Print the most likely prediction
        best_prediction = predictions[0]
        print(f"\nMost likely animal: {best_prediction[0].upper()}")
        print(f"   Confidence: {best_prediction[1]:.2%}")
    else:
        print("Failed to make prediction")

def list_available_models():
    """List all available trained model files"""
    print("Available trained models:")
    model_files = [f for f in os.listdir('.') if f.endswith('.pth')]
    if model_files:
        for model_file in sorted(model_files):
            print(f"  - {model_file}")
    else:
        print("  No model files found in current directory")

if __name__ == "__main__":
    # If no arguments provided, show help and available models
    if len(sys.argv) == 1:
        print("Animal Classifier - Inference Script")
        print("=" * 40)
        list_available_models()
        print("\nUsage examples:")
        print("  python animal_classifier.py --image cat.jpg --model resnet50")
        print("  python animal_classifier.py --image dog.jpg --model vgg16 --fold 3")
        print("  python animal_classifier.py --image bird.jpg --model inceptionv3 --top-k 5")
        print("\nFor full help: python animal_classifier.py --help")
    else:
        main() 