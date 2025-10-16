# FastAPI Road Segmentation with File Saving

This FastAPI application provides road segmentation services with automatic file saving functionality.

## Features

### File Management
- **Upload Images**: Automatically saves uploaded images to `upload_images/` folder
- **Prediction Results**: Saves segmentation results to `results_vis/` folder
- **File Naming**: Uses timestamps and unique IDs for file organization

### API Endpoints

#### 1. Model Management
- `POST /initialize_model` - Initialize the segmentation model
- `GET /model_info` - Get current model information
- `GET /health` - Health check endpoint

#### 2. Prediction Services
- `POST /predict` - Perform segmentation and return JSON results + save files
- `POST /predict_with_visualization` - Perform segmentation and return overlay image + save files

#### 3. File Management
- `GET /list_files` - List all uploaded images and prediction results
- `GET /download/{file_type}/{filename}` - Download specific files

### File Structure

When you upload an image and run prediction, the following files are automatically created:

```
upload_images/
└── 20241015_175230_abc12345.jpg  # Original uploaded image

results_vis/
├── 20241015_175230_image_segmentation_abc12345.png  # Binary segmentation mask
├── 20241015_175230_image_probability_abc12345.png   # Probability heatmap
└── 20241015_175230_image_overlay_abc12345.png       # Overlay visualization
```

### Usage Example

1. **Start the server:**
```bash
cd /path/to/Gistda-Sementi-Segmentation
python fastapi_segmentation.py
```

2. **Initialize the model:**
```bash
curl -X POST "http://localhost:8000/initialize_model" \
  -F "model_name=ConvNeXt_UPerNet_DGCN_MTL" \
  -F "dataset_name=DeepGlobe" \
  -F "experiment_name=U_net_round21" \
  -F "model_path=./Experiments/U_net_round21/model_best.pth.tar"
```

3. **Upload image and get prediction:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@your_road_image.jpg"
```

4. **List saved files:**
```bash
curl "http://localhost:8000/list_files"
```

5. **Download a specific file:**
```bash
curl "http://localhost:8000/download/results_vis/20241015_175230_image_overlay_abc12345.png" \
  --output downloaded_overlay.png
```

### Testing

Run the test script to verify functionality:
```bash
python test_api_with_files.py
```

This will:
1. Create a test image
2. Initialize the model
3. Test prediction with file saving
4. Test visualization with file saving
5. List all saved files
6. Show a summary of results

### File Naming Convention

Files are saved with the following naming pattern:
- **Format**: `{timestamp}_{original_name}_{type}_{unique_id}.{extension}`
- **Timestamp**: `YYYYMMDD_HHMMSS`
- **Types**: `segmentation`, `probability`, `overlay`
- **Unique ID**: 8-character hexadecimal string

### API Response with File Information

The `/predict` endpoint returns JSON with file paths:

```json
{
  "filename": "road_image.jpg",
  "uploaded_image_path": "upload_images/20241015_175230_abc12345.jpg",
  "input_shape": [512, 512],
  "output_shape": [512, 512],
  "segmentation_mask": [[0, 1, 1, ...], ...],
  "road_probability": [[0.1, 0.9, 0.8, ...], ...],
  "saved_files": {
    "segmentation_path": "results_vis/20241015_175230_road_image_segmentation_abc12345.png",
    "probability_path": "results_vis/20241015_175230_road_image_probability_abc12345.png",
    "overlay_path": "results_vis/20241015_175230_road_image_overlay_abc12345.png"
  }
}
```

### Requirements

Make sure you have all dependencies installed:
```bash
pip install fastapi uvicorn python-multipart torch torchvision opencv-python pillow numpy requests
```