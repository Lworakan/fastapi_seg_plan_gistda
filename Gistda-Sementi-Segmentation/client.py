#!/usr/bin/env python3
"""
FastAPI Client for Road Segmentation
Usage examples:
1. Initialize model: python client.py --init --model ConvNeXt_UPerNet_DGCN_MTL --dataset DeepGlobe --experiment U_net_round21
2. Predict: python client.py --predict --image path/to/image.jpg
3. Predict with visualization: python client.py --predict-viz --image path/to/image.jpg
"""

import requests
import argparse
import os
import json
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

class SegmentationClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def health_check(self):
        """Check if the API is running"""
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.status_code == 200, response.json()
        except requests.exceptions.RequestException as e:
            return False, str(e)
    
    def initialize_model(self, model_name, dataset_name, experiment_name, model_path=None):
        """Initialize the model on the server"""
        data = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "experiment_name": experiment_name
        }
        if model_path:
            data["model_path"] = model_path
        
        try:
            response = requests.post(f"{self.base_url}/initialize_model", data=data)
            return response.status_code == 200, response.json()
        except requests.exceptions.RequestException as e:
            return False, {"error": str(e)}
    
    def predict(self, image_path):
        """Get segmentation prediction for an image"""
        if not os.path.exists(image_path):
            return False, {"error": f"Image file not found: {image_path}"}
        
        try:
            with open(image_path, 'rb') as f:
                files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
                response = requests.post(f"{self.base_url}/predict", files=files)
            return response.status_code == 200, response.json()
        except requests.exceptions.RequestException as e:
            return False, {"error": str(e)}
    
    def predict_with_visualization(self, image_path, output_path=None):
        """Get segmentation prediction with visualization"""
        if not os.path.exists(image_path):
            return False, f"Image file not found: {image_path}"
        
        try:
            with open(image_path, 'rb') as f:
                files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
                response = requests.post(f"{self.base_url}/predict_with_visualization", files=files)
            
            if response.status_code == 200:
                if output_path is None:
                    output_path = f"segmentation_{os.path.basename(image_path)}"
                
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                return True, f"Visualization saved to: {output_path}"
            else:
                return False, response.json()
        except requests.exceptions.RequestException as e:
            return False, {"error": str(e)}
    
    def get_model_info(self):
        """Get current model information"""
        try:
            response = requests.get(f"{self.base_url}/model_info")
            return response.status_code == 200, response.json()
        except requests.exceptions.RequestException as e:
            return False, {"error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="FastAPI Road Segmentation Client")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--health", action="store_true", help="Check API health")
    parser.add_argument("--init", action="store_true", help="Initialize model")
    parser.add_argument("--predict", action="store_true", help="Predict segmentation")
    parser.add_argument("--predict-viz", action="store_true", help="Predict with visualization")
    parser.add_argument("--info", action="store_true", help="Get model info")
    
    # Model initialization parameters
    parser.add_argument("-m", "--model", default="ConvNeXt_UPerNet_DGCN_MTL", help="Model name")
    parser.add_argument("-d", "--dataset", default="DeepGlobe", help="Dataset name")
    parser.add_argument("-e", "--experiment", help="Experiment name")
    parser.add_argument("-r", "--model-path", help="Custom model path")
    
    # Prediction parameters
    parser.add_argument("-i", "--image", help="Image path for prediction")
    parser.add_argument("-o", "--output", help="Output path for visualization")
    
    args = parser.parse_args()
    
    client = SegmentationClient(args.url)
    
    if args.health:
        success, result = client.health_check()
        print("Health Check:", "✓ Healthy" if success else "✗ Unhealthy")
        print(json.dumps(result, indent=2))
    
    elif args.init:
        if not args.experiment:
            print("Error: --experiment is required for model initialization")
            return
        
        print(f"Initializing model: {args.model}")
        print(f"Dataset: {args.dataset}")
        print(f"Experiment: {args.experiment}")
        if args.model_path:
            print(f"Model path: {args.model_path}")
        
        success, result = client.initialize_model(
            args.model, args.dataset, args.experiment, args.model_path
        )
        
        if success:
            print("✓ Model initialized successfully")
            print(json.dumps(result, indent=2))
        else:
            print("✗ Model initialization failed")
            print(json.dumps(result, indent=2))
    
    elif args.predict:
        if not args.image:
            print("Error: --image is required for prediction")
            return
        
        print(f"Predicting segmentation for: {args.image}")
        success, result = client.predict(args.image)
        
        if success:
            print("✓ Prediction successful")
            print(f"Input shape: {result['input_shape']}")
            print(f"Output shape: {result['output_shape']}")
            print("Segmentation mask shape:", np.array(result['segmentation_mask']).shape)
            print("Road probability shape:", np.array(result['road_probability']).shape)
        else:
            print("✗ Prediction failed")
            print(json.dumps(result, indent=2))
    
    elif args.predict_viz:
        if not args.image:
            print("Error: --image is required for prediction with visualization")
            return
        
        print(f"Predicting with visualization for: {args.image}")
        success, result = client.predict_with_visualization(args.image, args.output)
        
        if success:
            print("✓ Prediction with visualization successful")
            print(result)
        else:
            print("✗ Prediction with visualization failed")
            print(json.dumps(result, indent=2))
    
    elif args.info:
        success, result = client.get_model_info()
        if success:
            print("Model Information:")
            print(json.dumps(result, indent=2))
        else:
            print("✗ Failed to get model info")
            print(json.dumps(result, indent=2))
    
    else:
        print("Please specify an action: --health, --init, --predict, --predict-viz, or --info")
        print("Use --help for more information")

if __name__ == "__main__":
    main()