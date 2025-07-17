#!/usr/bin/env python3
"""
Animal Classifier Web Application

A Flask web app for classifying animal images using trained models.
Users can upload images through a web interface and get classification results.
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
import io
import base64
from datetime import datetime

# Import animal classes from the original script
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

app = Flask(__name__)
app.secret_key = 'animal_classifier_secret_key_2024'

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables for model caching
cached_models = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def allowed_file(filename):
    """Check if uploaded file has allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_transforms(model_name):
    """Get the appropriate transforms for each model"""
    if model_name.lower() == 'inceptionv3':
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    return transform

def load_model(model_name, model_path):
    """Load and cache models"""
    cache_key = f"{model_name}_{model_path}"
    
    if cache_key in cached_models:
        return cached_models[cache_key]
    
    model_name = model_name.lower()
    
    try:
        if model_name == 'resnet50':
            model = models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        elif model_name == 'vgg16':
            model = models.vgg16(weights=None)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, NUM_CLASSES)
        elif model_name == 'inceptionv3':
            model = models.inception_v3(weights=None)
            model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
            model.aux_logits = False
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Load the trained weights
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        # Cache the model
        cached_models[cache_key] = model
        return model
        
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None

def classify_image(image, model_name, model_path, top_k=5):
    """Classify an image and return predictions"""
    try:
        # Load model
        model = load_model(model_name, model_path)
        if model is None:
            return None
        
        # Get transforms and preprocess image
        transform = get_transforms(model_name)
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
            predictions.append({
                'class': class_name,
                'confidence': confidence,
                'percentage': f"{confidence:.1%}"
            })
        
        return predictions
    
    except Exception as e:
        print(f"Error classifying image: {e}")
        return None

def get_available_models():
    """Get list of available model files"""
    model_files = {}
    
    for filename in os.listdir('.'):
        if filename.endswith('.pth'):
            if filename.startswith('resnet50_'):
                if 'resnet50' not in model_files:
                    model_files['resnet50'] = []
                model_files['resnet50'].append(filename)
            elif filename.startswith('vgg16_'):
                if 'vgg16' not in model_files:
                    model_files['vgg16'] = []
                model_files['vgg16'].append(filename)
            elif filename.startswith('inceptionv3_'):
                if 'inceptionv3' not in model_files:
                    model_files['inceptionv3'] = []
                model_files['inceptionv3'].append(filename)
    
    return model_files

@app.route('/')
def index():
    """Main page"""
    available_models = get_available_models()
    return render_template('index.html', 
                         available_models=available_models, 
                         device=device.type)

@app.route('/classify', methods=['POST'])
def classify():
    """Handle image classification"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload an image file.'}), 400
        
        # Get model selection
        model_name = request.form.get('model', 'resnet50')
        model_file = request.form.get('model_file')
        
        if not model_file:
            return jsonify({'error': 'No model file selected'}), 400
        
        if not os.path.exists(model_file):
            return jsonify({'error': f'Model file {model_file} not found'}), 400
        
        # Process the image
        image = Image.open(file.stream).convert('RGB')
        
        # Convert image to base64 for display
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='JPEG', quality=85)
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        
        # Classify the image
        predictions = classify_image(image, model_name, model_file)
        
        if predictions is None:
            return jsonify({'error': 'Classification failed'}), 500
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'image': img_str,
            'model': model_name,
            'model_file': model_file,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/models')
def get_models():
    """API endpoint to get available models"""
    return jsonify(get_available_models())

if __name__ == '__main__':
    print(f"Animal Classifier Web App")
    print(f"Device: {device}")
    print(f"Available models: {get_available_models()}")
    print("Starting server at http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000) 