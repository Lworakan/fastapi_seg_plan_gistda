from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
import cv2
from skimage import io
from PIL import Image
import io as python_io
import tempfile
from typing import Optional
import uuid
import base64
import datetime
from pathlib import Path

# Add project paths to system path
sys.path.append('./Models')
sys.path.append('./Tools')

# Import models
from ConvNeXt_UPerNet_DGCN_MTL import ConvNeXt_UPerNet_DGCN_MTL
from DeepLabV3_MTL_Adapter import DeepLabV3_MTL_Adapter

# Import project modules
import DatasetUtility
import Losses
import util
import viz_util

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Create necessary directories
UPLOAD_DIR = Path("upload_images")
RESULTS_DIR = Path("results_vis")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="Road Segmentation API",
    description="FastAPI service for road segmentation using ConvNeXt UPerNet model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and configuration
model = None
cfg = None
segmentation_loss = None
orientation_loss = None
nGPUs = None

class SegmentationPredictor:
    def __init__(self, model_name: str, dataset_name: str, experiment_name: str, model_path: str):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.experiment_name = experiment_name
        self.model_path = model_path
        self.model = None
        self.cfg = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._load_config()
        self._initialize_model()
        self._load_weights()
    
    def _load_config(self):
        """Load configuration from cfg.json"""
        with open("cfg.json", 'r') as f:
            self.cfg = json.load(f)
    
    def _initialize_model(self):
        """Initialize the model based on model name"""
        if self.model_name == "ConvNeXt_UPerNet_DGCN_MTL":
            # ConvNeXt model - use Base architecture by default
            self.model = ConvNeXt_UPerNet_DGCN_MTL("Base")
        elif self.model_name == "DeepLabV3_MTL_Adapter":
            # DeepLabV3 model with default parameters
            self.model = DeepLabV3_MTL_Adapter()
        else:
            raise ValueError(f"Model {self.model_name} not supported. Available models: ConvNeXt_UPerNet_DGCN_MTL, DeepLabV3_MTL_Adapter")
        
        # Move model to appropriate device
        if torch.cuda.is_available():
            nGPUs = torch.cuda.device_count()
            if nGPUs == 1:
                self.model.cuda()
                self.device = torch.device("cuda")
            else:
                self.model = nn.DataParallel(self.model).cuda()
                self.device = torch.device("cuda")
        else:
            print("CUDA not available, using CPU")
            self.device = torch.device("cpu")
    
    def _load_weights(self):
        """Load pre-trained weights"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model weights not found at {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        print(f"Model weights loaded from {self.model_path}")
    
    def preprocess_image(self, image_array: np.ndarray) -> torch.Tensor:
        """Preprocess input image for model inference"""
        # Convert BGR to RGB if needed
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        
        # Get normalization parameters from config
        if self.dataset_name in self.cfg["Datasets"]:
            mean = eval(self.cfg["Datasets"][self.dataset_name]["mean"])
            std = eval(self.cfg["Datasets"][self.dataset_name]["std"])
        else:
            # Default normalization
            mean = [70.95, 71.16, 71.31]
            std = [34.00, 35.18, 36.40]
        
        # Resize to crop size
        crop_size = self.cfg["validation_settings"]["crop_size"]
        if self.dataset_name == "Spacenet":
            crop_size = self.cfg["validation_settings"]["spacenet_crop_size"]
        
        image_resized = cv2.resize(image_array, (crop_size, crop_size))
        
        # Normalize
        image_normalized = image_resized.astype(np.float32)
        image_normalized = (image_normalized - np.array(mean)) / np.array(std)
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image_normalized.transpose(2, 0, 1)).float()
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
        return image_tensor.to(self.device)
    
    def predict(self, image_array: np.ndarray) -> dict:
        """Perform segmentation prediction on input image"""
        with torch.no_grad():
            # Preprocess image
            image_tensor = self.preprocess_image(image_array)
            
            # Forward pass
            predictions = self.model(image_tensor)
            predicted_road = predictions[0][-1]  # Get the final scale prediction
            predicted_orientation = predictions[1][-1] if len(predictions) > 1 else None
            
            # Get segmentation mask
            _, predicted_road_mask = torch.max(predicted_road, 1)
            segmentation_mask = predicted_road_mask.cpu().numpy().squeeze()
            
            # Get probability maps
            road_probabilities = F.softmax(predicted_road, dim=1).cpu().numpy()
            road_prob_map = road_probabilities[0, 1, :, :]  # Road class probability
            
            # Get orientation predictions if available
            orientation_mask = None
            if predicted_orientation is not None:
                _, predicted_orientation_mask = torch.max(predicted_orientation, 1)
                orientation_mask = predicted_orientation_mask.cpu().numpy().squeeze()
            
            return {
                "segmentation_mask": segmentation_mask.astype(np.uint8),
                "road_probability": road_prob_map.astype(np.float32),
                "orientation_mask": orientation_mask.astype(np.uint8) if orientation_mask is not None else None,
                "input_shape": image_array.shape[:2],
                "output_shape": segmentation_mask.shape
            }

# Global predictor instance
predictor = None

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup with default parameters"""
    global predictor
    # You can set default values or leave it to be initialized on first prediction
    print("FastAPI segmentation service started")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Road Segmentation API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "cuda_available": torch.cuda.is_available()}

@app.post("/initialize_model")
async def initialize_model(
    model_name: str = Form(..., description="Model name (e.g., ConvNeXt_UPerNet_DGCN_MTL)"),
    dataset_name: str = Form(..., description="Dataset name (e.g., DeepGlobe, MassachusettsRoads)"),
    experiment_name: str = Form(..., description="Experiment name"),
    model_path: Optional[str] = Form(None, description="Custom model path (optional)")
):
    """Initialize the model with specified parameters"""
    global predictor
    
    try:
        # Set default model path if not provided
        if model_path is None:
            model_path = f"./Experiments/{experiment_name}/model_best.pth.tar"
        
        predictor = SegmentationPredictor(
            model_name=model_name,
            dataset_name=dataset_name,
            experiment_name=experiment_name,
            model_path=model_path
        )
        
        return {
            "message": "Model initialized successfully",
            "model_name": model_name,
            "dataset_name": dataset_name,
            "experiment_name": experiment_name,
            "model_path": model_path
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize model: {str(e)}")

def save_uploaded_image(file_content: bytes, filename: str) -> str:
    """Save uploaded image to upload_images folder"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_extension = Path(filename).suffix
    unique_filename = f"{timestamp}_{uuid.uuid4().hex[:8]}{file_extension}"
    file_path = UPLOAD_DIR / unique_filename
    
    with open(file_path, "wb") as f:
        f.write(file_content)
    
    return str(file_path)

def save_prediction_results(image_array: np.ndarray, result: dict, original_filename: str) -> dict:
    """Save prediction results to results_vis folder"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = Path(original_filename).stem
    unique_id = uuid.uuid4().hex[:8]
    
    # Create file paths
    segmentation_path = RESULTS_DIR / f"{timestamp}_{base_name}_segmentation_{unique_id}.png"
    probability_path = RESULTS_DIR / f"{timestamp}_{base_name}_probability_{unique_id}.png"
    overlay_path = RESULTS_DIR / f"{timestamp}_{base_name}_overlay_{unique_id}.png"
    
    segmentation_mask = result["segmentation_mask"]
    road_probability = result["road_probability"]
    
    # Save segmentation mask (binary mask)
    segmentation_image = (segmentation_mask * 255).astype(np.uint8)
    cv2.imwrite(str(segmentation_path), segmentation_image)
    
    # Save probability map (heatmap)
    probability_image = (road_probability * 255).astype(np.uint8)
    probability_colored = cv2.applyColorMap(probability_image, cv2.COLORMAP_JET)
    cv2.imwrite(str(probability_path), probability_colored)
    
    # Create and save overlay
    colored_mask = np.zeros((segmentation_mask.shape[0], segmentation_mask.shape[1], 3), dtype=np.uint8)
    colored_mask[segmentation_mask == 1] = [0, 255, 0]  # Green for roads
    
    # Resize original image to match output size if needed
    if image_array.shape[:2] != segmentation_mask.shape:
        image_resized = cv2.resize(image_array, (segmentation_mask.shape[1], segmentation_mask.shape[0]))
    else:
        image_resized = image_array
    
    # Create overlay
    overlay = cv2.addWeighted(image_resized, 0.7, colored_mask, 0.3, 0)
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    return {
        "segmentation_path": str(segmentation_path),
        "probability_path": str(probability_path),
        "overlay_path": str(overlay_path)
    }

@app.post("/predict")
async def predict_segmentation(file: UploadFile = File(...)):
    """Perform segmentation on uploaded image"""
    global predictor
    
    if predictor is None:
        raise HTTPException(status_code=400, detail="Model not initialized. Please call /initialize_model first.")
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        contents = await file.read()
        
        # Save uploaded image
        uploaded_image_path = save_uploaded_image(contents, file.filename)
        
        image = Image.open(python_io.BytesIO(contents))
        image_array = np.array(image)
        
        # Perform prediction
        result = predictor.predict(image_array)
        
        # Save prediction results
        saved_files = save_prediction_results(image_array, result, file.filename)
        
        # Convert numpy arrays to lists for JSON serialization
        response_data = {
            "filename": file.filename,
            "uploaded_image_path": uploaded_image_path,
            "input_shape": result["input_shape"],
            "output_shape": result["output_shape"],
            "segmentation_mask": result["segmentation_mask"].tolist(),
            "road_probability": result["road_probability"].tolist(),
            "saved_files": saved_files
        }
        
        if result["orientation_mask"] is not None:
            response_data["orientation_mask"] = result["orientation_mask"].tolist()
        
        return JSONResponse(content=response_data)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_with_visualization")
async def predict_with_visualization(file: UploadFile = File(...)):
    """Perform segmentation and return visualization"""
    global predictor
    
    if predictor is None:
        raise HTTPException(status_code=400, detail="Model not initialized. Please call /initialize_model first.")
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        contents = await file.read()
        
        # Save uploaded image
        uploaded_image_path = save_uploaded_image(contents, file.filename)
        
        image = Image.open(python_io.BytesIO(contents))
        image_array = np.array(image)
        
        # Perform prediction
        result = predictor.predict(image_array)
        
        # Save prediction results to files
        saved_files = save_prediction_results(image_array, result, file.filename)
        
        # Return the overlay visualization file directly
        return FileResponse(
            saved_files["overlay_path"],
            media_type='image/png',
            filename=f"segmentation_overlay_{file.filename}",
            headers={
                "X-Uploaded-Image-Path": uploaded_image_path,
                "X-Segmentation-Path": saved_files["segmentation_path"],
                "X-Probability-Path": saved_files["probability_path"]
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")

@app.get("/model_info")
async def get_model_info():
    """Get current model information"""
    global predictor
    
    if predictor is None:
        return {"message": "No model initialized"}
    
    return {
        "model_name": predictor.model_name,
        "dataset_name": predictor.dataset_name,
        "experiment_name": predictor.experiment_name,
        "model_path": predictor.model_path,
        "device": str(predictor.device),
        "cuda_available": torch.cuda.is_available()
    }

@app.get("/list_files")
async def list_saved_files():
    """List uploaded images and prediction results"""
    upload_files = list(UPLOAD_DIR.glob("*")) if UPLOAD_DIR.exists() else []
    result_files = list(RESULTS_DIR.glob("*")) if RESULTS_DIR.exists() else []
    
    return {
        "upload_images": [str(f.name) for f in upload_files if f.is_file()],
        "prediction_results": [str(f.name) for f in result_files if f.is_file()],
        "upload_count": len([f for f in upload_files if f.is_file()]),
        "results_count": len([f for f in result_files if f.is_file()])
    }

@app.get("/download/{file_type}/{filename}")
async def download_file(file_type: str, filename: str):
    """Download saved files (upload_images or results_vis)"""
    if file_type == "upload_images":
        file_path = UPLOAD_DIR / filename
    elif file_type == "results_vis":
        file_path = RESULTS_DIR / filename
    else:
        raise HTTPException(status_code=400, detail="Invalid file_type. Use 'upload_images' or 'results_vis'")
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)