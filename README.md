# FastAPI Road Segmentation & Path Planning System

A comprehensive FastAPI-based system for road segmentation using deep learning models and intelligent path planning for flood navigation scenarios.

## Overview

This project combines two powerful FastAPI services:

1. **Road Segmentation API** - Deep learning-based road segmentation using ConvNeXt UPerNet and DeepLabV3 models
2. **Path Planning API** - Multi-goal path planning with real-world mapping and visualization

## Project Structure

```
fastapi_seg_plan_gistda/
├── LICENSE
│
├── Gistda-Sementi-Segmentation/          # Road Segmentation Service
│   ├── fastapi_segmentation.py           # Main FastAPI segmentation server
│   ├── start_server.sh                   # Server startup script
│   ├── fastapi_requirements.txt          # FastAPI dependencies
│   ├── requirements.txt                  # ML model dependencies
│   ├── cfg.json                          # Configuration file
│   ├── client.py                         # API client example
│   ├── test_api.py                       # API testing script
│   │
│   ├── Models/                           # Deep learning models
│   │   ├── ConvNeXt_UPerNet_DGCN_MTL.py # ConvNeXt UPerNet model
│   │   └── DeepLabV3_MTL_Adapter.py     # DeepLabV3 model
│   │
│   ├── Tools/                            # Utility modules
│   │   ├── DatasetUtility.py
│   │   ├── ImageStatistics.py
│   │   ├── Losses.py
│   │   ├── LineConversion.py
│   │   └── viz_util.py
│   │
│   ├── Datasets/                         # Dataset configurations
│   │   ├── DeepGlobe/
│   │   │   ├── class_dict.csv
│   │   │   └── metadata.csv
│   │   └── MassachusettsRoads/
│   │       ├── label_class_dict.csv
│   │       └── metadata.csv
│   │
│   ├── Experiments/                      # Model experiments
│   │   └── U_net_round21/
│   │
│   ├── upload_images/                    # Uploaded images storage
│   ├── results_vis/                      # Segmentation results
│   └── docs/                             # Documentation
│
└── FloodNav-Path-Planning/               # Path Planning Service
    ├── fastapi_path_planning.py          # Main FastAPI path planning server
    ├── start_fastapi_server.sh           # Server startup script
    ├── requirements_fastapi.txt          # FastAPI dependencies
    ├── test_fastapi_client.py            # API testing script
    ├── example_request_data.json         # Example request format
    │
    ├── path_planning/                    # Path planning modules
    │   ├── __init__.py
    │   ├── A_star.py                     # A* algorithm
    │   ├── best_first.py                 # Best-first search
    │   ├── breadth_first.py              # BFS algorithm
    │   ├── main.py                       # Main planning logic
    │   ├── path_planning_manager.py      # Manager class
    │   ├── path_visualizer.py            # Visualization utilities
    │   └── path_width_analyzer.py        # Path width analysis
    │
    ├── preprocess/                       # Preprocessing utilities
    │   ├── clicked_pixel.py
    │   └── create_grid.py
    │
    ├── resource/                         # Resource files
    │   ├── map.npy                       # Grid map data
    │   └── map2.npy
    │
    ├── results/                          # Path planning results
    │   └── path_planning_YYYYMMDD_HHMMSS/
    │       └── metadata.json
    │
    ├── output_plots/                     # Generated visualizations
    │
    └── working_curl_commands.sh          # Example curl commands
```

## Quick Start

### Prerequisites

- Python 3.12
- CUDA-capable GPU (optional, for faster segmentation)
- Conda or Miniconda ([Download here](https://docs.conda.io/en/latest/miniconda.html))

### Installation

#### Option 1: Using Conda Environment File (Recommended)

This method creates a complete environment with all dependencies from your exported configuration:

```bash
# Clone the repository
cd /path/to/fastapi_seg_plan_gistda

# Create conda environment from file
conda env create -f environment.yml

# Activate the environment
conda activate fastapi_seg_plan

# Verify installation
python --version  # Should show Python 3.12.9
```

#### Option 2: Manual Installation

##### 1. Road Segmentation Service

```bash
cd Gistda-Sementi-Segmentation

# Create and activate conda environment
conda create -n fastapi_seg_plan python=3.12
conda activate fastapi_seg_plan

# Install dependencies
pip install -r fastapi_requirements.txt
pip install -r requirements.txt

# Start the server
bash start_server.sh
# Or manually:
# uvicorn fastapi_segmentation:app --host 0.0.0.0 --port 8000
```

##### 2. Path Planning Service

Both services can use the same conda environment:

```bash
# Make sure you're in the conda environment
conda activate fastapi_seg_plan

# Navigate to path planning directory
cd FloodNav-Path-Planning

# Start the server
bash start_fastapi_server.sh
# Or manually:
# uvicorn fastapi_path_planning:app --host 0.0.0.0 --port 8000 --reload
```

### Running Both Services

If you need to run both services simultaneously, use different ports:

```bash
# Terminal 1 - Segmentation Service
conda activate fastapi_seg_plan
cd Gistda-Sementi-Segmentation
uvicorn fastapi_segmentation:app --host 0.0.0.0 --port 8000

# Terminal 2 - Path Planning Service
conda activate fastapi_seg_plan
cd FloodNav-Path-Planning
uvicorn fastapi_path_planning:app --host 0.0.0.0 --port 8001
```

## 📡 API Documentation

### Road Segmentation API (Port 8000)

#### Base URL: `http://localhost:8000`

#### Endpoints:

##### 1. Health Check
```http
GET /health
```
Response:
```json
{
  "status": "healthy",
  "cuda_available": true
}
```

##### 2. Initialize Model
```http
POST /initialize_model
Content-Type: multipart/form-data
```

Parameters:
- `model_name` (required): `ConvNeXt_UPerNet_DGCN_MTL` or `DeepLabV3_MTL_Adapter`
- `dataset_name` (required): `DeepGlobe` or `MassachusettsRoads`
- `experiment_name` (required): Your experiment name
- `model_path` (optional): Custom model weight path

Example:
```bash
curl -X POST "http://localhost:8000/initialize_model" \
  -F "model_name=ConvNeXt_UPerNet_DGCN_MTL" \
  -F "dataset_name=DeepGlobe" \
  -F "experiment_name=U_net_round21"
```

```bash
curl -X POST "http://localhost:8000/plan_path" \                                                   ─╯
  -F 'request_data={"start_point":{"x":454,"y":368},"goal_points":[{"x":852,"y":460}],"scale_pix_to_m":0.05,"k_top_paths":3,"hausdorff_tolerance":10.0}' \
  -F "segmentation_image=@resource/map.png" \
  -F "real_world_image=@resource/satellite.png"
```
##### 3. Predict Segmentation
```http
POST /predict
Content-Type: multipart/form-data
```

Parameters:
- `file` (required): Image file to segment

Response:
```json
{
  "filename": "satellite_image.png",
  "uploaded_image_path": "upload_images/20251016_120000_abc123.png",
  "input_shape": [1024, 1024],
  "output_shape": [1024, 1024],
  "segmentation_mask": [[0, 1, 1, ...], ...],
  "road_probability": [[0.1, 0.9, 0.95, ...], ...],
  "saved_files": {
    "segmentation_path": "results_vis/..._segmentation.png",
    "probability_path": "results_vis/..._probability.png",
    "overlay_path": "results_vis/..._overlay.png"
  }
}
```

##### 4. Get Model Info
```http
GET /model_info
```

##### 5. List Saved Files
```http
GET /list_files
```

##### 6. Download Files
```http
GET /download/{file_type}/{filename}
```
- `file_type`: `upload_images` or `results_vis`
- `filename`: Name of the file

### Path Planning API (Port 8000)

#### Base URL: `http://localhost:8000`

#### Endpoints:

##### 1. Health Check
```http
GET /health
```

##### 2. Plan Path
```http
POST /plan_path
Content-Type: multipart/form-data
```

Parameters:
- `request_data` (required): JSON string with planning parameters
- `segmentation_image` (required): Binary segmentation image (white=roads, black=obstacles)
- `real_world_image` (optional): Satellite/aerial image for overlay visualization

Request data format:
```json
{
  "start_point": {"x": 100, "y": 150},
  "goal_points": [
    {"x": 500, "y": 400},
    {"x": 800, "y": 600}
  ],
  "scale_pix_to_m": 0.05,
  "k_top_paths": 3,
  "hausdorff_tolerance": 10.0
}
```

Response:
```json
{
  "success": true,
  "message": "Path planning completed successfully",
  "total_paths_found": 6,
  "total_combinations_tested": 12,
  "results": [
    {
      "path_id": 1,
      "algorithm": "A*",
      "distance_pixels": 1234.5,
      "distance_meters": 61.725,
      "path_coordinates": [[100, 150], [101, 151], ...],
      "width_stats": {
        "min_width": 5.2,
        "max_width": 15.8,
        "mean_width": 10.3
      }
    }
  ],
  "algorithm_performance": {},
  "visualization_images": {
    "grid_paths": "base64_encoded_image...",
    "performance": "base64_encoded_image...",
    "width_analysis": "base64_encoded_image..."
  },
  "results_directory": "results/path_planning_20251016_120000"
}
```

##### 3. Get Available Algorithms
```http
GET /algorithms
```

##### 4. Validate Coordinates
```http
POST /validate_coordinates
Content-Type: multipart/form-data
```

##  Usage Examples

### Python Client Example (Segmentation)

```python
import requests

# Initialize model
init_response = requests.post(
    "http://localhost:8000/initialize_model",
    data={
        "model_name": "ConvNeXt_UPerNet_DGCN_MTL",
        "dataset_name": "DeepGlobe",
        "experiment_name": "U_net_round21"
    }
)

# Predict segmentation
with open("satellite_image.png", "rb") as f:
    files = {"file": f}
    response = requests.post(
        "http://localhost:8000/predict",
        files=files
    )
    result = response.json()
    print(f"Segmentation saved to: {result['saved_files']['segmentation_path']}")
```

### Python Client Example (Path Planning)

```python
import requests
import json

# Prepare request data
request_data = {
    "start_point": {"x": 100, "y": 150},
    "goal_points": [
        {"x": 500, "y": 400},
        {"x": 800, "y": 600}
    ],
    "scale_pix_to_m": 0.05,
    "k_top_paths": 3,
    "hausdorff_tolerance": 10.0
}

# Send request
with open("segmentation_mask.png", "rb") as seg_img:
    with open("satellite_image.png", "rb") as real_img:
        files = {
            "segmentation_image": seg_img,
            "real_world_image": real_img
        }
        data = {"request_data": json.dumps(request_data)}
        
        response = requests.post(
            "http://localhost:8000/plan_path",
            data=data,
            files=files
        )
        result = response.json()
        print(f"Found {result['total_paths_found']} paths")
        print(f"Results saved to: {result['results_directory']}")
```

##  Features

### Road Segmentation Service
-  Multiple deep learning models (ConvNeXt UPerNet, DeepLabV3)
-  Real-time road segmentation
-  Probability maps and binary masks
-  Automatic result visualization and storage
-  File management (upload, download, list)
-  GPU acceleration support

### Path Planning Service
-  Multiple path planning algorithms (A*, Best-First, BFS)
-  Multi-goal path optimization
-  Path width analysis
-  Real-world coordinate mapping
-  Interactive visualizations
-  Hausdorff distance-based path deduplication
-  Algorithm performance comparison

##  Output Files

### Segmentation Results
- `upload_images/`: Original uploaded images
- `results_vis/`: 
  - `*_segmentation.png`: Binary segmentation mask
  - `*_probability.png`: Probability heatmap
  - `*_overlay.png`: Overlay on original image

### Path Planning Results
- `results/path_planning_YYYYMMDD_HHMMSS/`:
  - `grid_paths.png`: Grid-based path visualization
  - `performance.png`: Algorithm performance comparison
  - `width_analysis.png`: Path width analysis charts
  - `real_world_overlay.png`: Paths on satellite imagery
  - `metadata.json`: Planning metadata

##  Testing

### Test Segmentation API
```bash
cd Gistda-Sementi-Segmentation
python test_api.py
```

### Test Path Planning API
```bash
cd FloodNav-Path-Planning
python test_fastapi_client.py
```

## 🌐 Interactive API Documentation

Both services provide interactive API documentation:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

##  Configuration

### Segmentation Configuration (`cfg.json`)
```json
{
  "Datasets": {
    "DeepGlobe": {
      "mean": "[70.95, 71.16, 71.31]",
      "std": "[34.00, 35.18, 36.40]"
    }
  },
  "validation_settings": {
    "crop_size": 1024
  }
}
```

### Path Planning Parameters
- `scale_pix_to_m`: Pixel to meter conversion (default: 0.05)
- `k_top_paths`: Number of top paths to return (default: 3)
- `hausdorff_tolerance`: Path similarity threshold (default: 10.0)

##  CORS Configuration

Both APIs are configured with permissive CORS settings for development. For production, update CORS settings in:
- `fastapi_segmentation.py`
- `fastapi_path_planning.py`

##  Debugging

Enable debug mode:
```python
# In fastapi_segmentation.py or fastapi_path_planning.py
uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="debug")
```

##  License

This project is licensed under the terms specified in the LICENSE file.

##  Contributing

Contributions are welcome! Please ensure:
1. Code follows PEP 8 style guidelines
2. All tests pass
3. Documentation is updated

## Support

For issues and questions:
1. Check the interactive API documentation at `/docs`
2. Review example scripts in the repository
3. Check debug output in terminal logs

##  Version History

- **v1.0.0** (2025-10-16): Initial release with road segmentation and path planning APIs

---

**Note**: Ensure you have the required model weights in the `Experiments/` directory before running the segmentation service.
