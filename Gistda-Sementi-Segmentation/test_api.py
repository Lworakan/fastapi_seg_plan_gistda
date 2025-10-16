#!/usr/bin/env python3
"""
Test script for FastAPI segmentation service
This script initializes the model and tests prediction functionality
"""

import requests
import json
import sys
import os
from PIL import Image
import numpy as np

# API base URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"Response: {response.json()}")
            return True
        else:
            print("❌ Health check failed")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def initialize_model(model_name="ConvNeXt_UPerNet_DGCN_MTL", 
                    dataset_name="DeepGlobe", 
                    experiment_name="U_net_round21",
                    model_path=None):
    """Initialize the segmentation model"""
    try:
        # If no model path provided, use the default pattern
        if model_path is None:
            model_path = f"./Experiments/{experiment_name}/model_best.pth.tar"
        
        data = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "experiment_name": experiment_name,
            "model_path": model_path
        }
        
        print(f"🔧 Initializing model with parameters:")
        print(f"   Model: {model_name}")
        print(f"   Dataset: {dataset_name}")
        print(f"   Experiment: {experiment_name}")
        print(f"   Model path: {model_path}")
        
        response = requests.post(f"{BASE_URL}/initialize_model", data=data)
        
        if response.status_code == 200:
            print("✅ Model initialized successfully")
            print(f"Response: {response.json()}")
            return True
        else:
            print("❌ Model initialization failed")
            print(f"Status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Model initialization error: {e}")
        return False

def get_model_info():
    """Get current model information"""
    try:
        response = requests.get(f"{BASE_URL}/model_info")
        if response.status_code == 200:
            print("📊 Model info:")
            info = response.json()
            for key, value in info.items():
                print(f"   {key}: {value}")
            return True
        else:
            print("❌ Failed to get model info")
            return False
    except Exception as e:
        print(f"❌ Model info error: {e}")
        return False

def create_test_image():
    """Create a simple test image"""
    # Create a simple RGB test image
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    image = Image.fromarray(test_image)
    test_path = "test_image.png"
    image.save(test_path)
    return test_path

def test_prediction(image_path=None):
    """Test prediction endpoint"""
    try:
        # Create test image if none provided
        if image_path is None:
            image_path = create_test_image()
            print(f"📸 Created test image: {image_path}")
        
        if not os.path.exists(image_path):
            print(f"❌ Image file not found: {image_path}")
            return False
        
        print(f"🔍 Testing prediction with image: {image_path}")
        
        with open(image_path, 'rb') as f:
            files = {'file': ('test.png', f, 'image/png')}
            response = requests.post(f"{BASE_URL}/predict", files=files)
        
        if response.status_code == 200:
            print("✅ Prediction successful")
            result = response.json()
            print(f"   Input shape: {result['input_shape']}")
            print(f"   Output shape: {result['output_shape']}")
            print(f"   Filename: {result['filename']}")
            return True
        else:
            print("❌ Prediction failed")
            print(f"Status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Testing FastAPI Segmentation Service")
    print("=" * 50)
    
    # Test 1: Health check
    if not test_health():
        print("❌ Server is not running. Please start the server first with: ./start_server.sh")
        return
    
    print()
    
    # Test 2: Initialize model
    # Using the model path you provided
    model_path = "/Users/worakanlasudee/Documents/GitHub/fastapi_seg_plan/Gistda-Sementi-Segmentation/Experiments/U_net_round21/model_best.pth.tar"
    
    if not initialize_model(model_path=model_path):
        print("❌ Failed to initialize model")
        return
    
    print()
    
    # Test 3: Get model info
    get_model_info()
    
    print()
    
    # Test 4: Test prediction
    test_prediction()
    
    print()
    print("🎉 All tests completed!")

if __name__ == "__main__":
    main()