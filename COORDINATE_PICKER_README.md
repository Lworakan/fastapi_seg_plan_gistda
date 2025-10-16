# Coordinate Picker Tools

Interactive tools to click on images and capture pixel coordinates for path planning.

## 📁 Available Tools

### 1. `click_coordinates.py` (Root directory)
Full-featured coordinate picker with advanced visualization.

**Features:**
- ✅ Visual crosshair cursor with live coordinates
- ✅ Color-coded points (Green=START, Red=GOAL)
- ✅ Connecting lines between points
- ✅ Coordinate labels on image
- ✅ Auto-save with timestamp
- ✅ FastAPI-ready JSON format

**Usage:**
```bash
python click_coordinates.py <image_path>
```

**Example:**
```bash
python click_coordinates.py FloodNav-Path-Planning/satellite.png
```

### 2. `get_coordinates.py` (FloodNav-Path-Planning directory)
Simplified coordinate picker with file dialog.

**Features:**
- ✅ GUI file picker (no command line arguments needed)
- ✅ Simple and intuitive
- ✅ Auto-save on exit
- ✅ FastAPI-ready format

**Usage:**
```bash
cd FloodNav-Path-Planning
python get_coordinates.py
```

## 🎮 Controls

Both tools use the same controls:

| Action | Control |
|--------|---------|
| Add point | Left Click |
| Remove last point | Right Click |
| Save coordinates | Press `s` |
| Clear all points | Press `c` |
| Quit | Press `q` or `ESC` |

## 📤 Output Format

Coordinates are saved to `coordinates_output/` directory in JSON format:

```json
{
  "image_path": "path/to/image.png",
  "image_size": {
    "width": 1024,
    "height": 1024
  },
  "timestamp": "20251016_120000",
  "start_point": {
    "x": 100,
    "y": 150
  },
  "goal_points": [
    {"x": 500, "y": 400},
    {"x": 800, "y": 600}
  ],
  "fastapi_request": {
    "start_point": {"x": 100, "y": 150},
    "goal_points": [
      {"x": 500, "y": 400},
      {"x": 800, "y": 600}
    ],
    "scale_pix_to_m": 0.05,
    "k_top_paths": 3,
    "hausdorff_tolerance": 10.0
  }
}
```

## 🔧 Integration with Path Planning API

The saved JSON can be directly used with the FastAPI path planning endpoint:

### Method 1: Using the saved JSON

```python
import requests
import json

# Load saved coordinates
with open('coordinates_output/coords_satellite_20251016_120000.json', 'r') as f:
    data = json.load(f)

# Use with FastAPI
with open('segmentation_mask.png', 'rb') as seg_img:
    with open('satellite_image.png', 'rb') as real_img:
        files = {
            'segmentation_image': seg_img,
            'real_world_image': real_img
        }
        request_data = {
            'request_data': json.dumps(data['fastapi_request'])
        }
        
        response = requests.post(
            'http://localhost:8000/plan_path',
            data=request_data,
            files=files
        )
        print(response.json())
```

### Method 2: Using curl

```bash
# Extract the coordinates from saved JSON
REQUEST_DATA=$(cat coordinates_output/coords_satellite_20251016_120000.json | jq -c '.fastapi_request')

curl -X POST "http://localhost:8000/plan_path" \
  -F "request_data=$REQUEST_DATA" \
  -F "segmentation_image=@segmentation_mask.png" \
  -F "real_world_image=@satellite_image.png"
```

## 🎯 Workflow Example

### Step 1: Pick coordinates on segmentation image
```bash
python click_coordinates.py FloodNav-Path-Planning/segmentation.png
```

1. Click on the **start position** (first click = green marker)
2. Click on **goal positions** (subsequent clicks = red markers)
3. Press `s` to save
4. Note the saved file path

### Step 2: Use coordinates with API

```python
import requests
import json

# Load the saved coordinates
with open('coordinates_output/coords_segmentation_20251016_120000.json', 'r') as f:
    coords = json.load(f)

# Send to path planning API
with open('FloodNav-Path-Planning/segmentation.png', 'rb') as seg:
    with open('FloodNav-Path-Planning/satellite.png', 'rb') as sat:
        response = requests.post(
            'http://localhost:8000/plan_path',
            data={'request_data': json.dumps(coords['fastapi_request'])},
            files={
                'segmentation_image': seg,
                'real_world_image': sat
            }
        )

result = response.json()
print(f"✅ Found {result['total_paths_found']} paths")
print(f"📁 Results: {result['results_directory']}")
```

## 📊 Coordinate System

```
(0,0) ──────────────────► x (width)
  │
  │
  │
  │
  ▼
  y (height)
```

- **Origin (0,0)**: Top-left corner
- **X-axis**: Increases to the right (width)
- **Y-axis**: Increases downward (height)
- **Max coordinates**: (width-1, height-1)

## 💡 Tips

1. **First click is always START**: The first point you click becomes the start position
2. **Subsequent clicks are GOALS**: All other points become goal waypoints
3. **Order matters**: Goals are visited in the order you click them
4. **Right-click to undo**: Remove the last point if you misclicked
5. **Save frequently**: Press `s` to save your progress
6. **Use crosshair**: Hover to see exact pixel coordinates before clicking

## 🐛 Troubleshooting

### Image doesn't open
```bash
# Check file exists
ls -l path/to/image.png

# Check file format
file path/to/image.png
```

### Coordinates out of bounds
The tools prevent clicking outside image bounds, but if you manually edit JSON:
- Ensure `0 <= x < image_width`
- Ensure `0 <= y < image_height`

### Dependencies missing
```bash
# Install required packages
pip install opencv-python numpy
```

## 📝 Notes

- Coordinates are saved as **integers** (pixel indices)
- Output directory `coordinates_output/` is created automatically
- Each save creates a new file with timestamp
- Files are named: `coords_<image_name>_<timestamp>.json`

## 🔗 Related Files

- **Path Planning API**: `FloodNav-Path-Planning/fastapi_path_planning.py`
- **Test Client**: `FloodNav-Path-Planning/test_fastapi_client.py`
- **Example Data**: `FloodNav-Path-Planning/example_request_data.json`
