# Example Request Data for FastAPI Path Planning

## Example 1: Basic Path Planning Request

```json
{
  "start_point": {
    "x": 454,
    "y": 368
  },
  "goal_points": [
    {
      "x": 852,
      "y": 460
    },
    {
      "x": 234,
      "y": 830
    },
    {
      "x": 302,
      "y": 990
    }
  ],
  "scale_pix_to_m": 0.05,
  "k_top_paths": 3,
  "hausdorff_tolerance": 10.0
}
```

## Example 2: Minimal Request (using defaults)

```json
{
  "start_point": {
    "x": 100,
    "y": 100
  },
  "goal_points": [
    {
      "x": 500,
      "y": 200
    },
    {
      "x": 300,
      "y": 600
    }
  ]
}
```

## Example 3: High Precision Request

```json
{
  "start_point": {
    "x": 150.5,
    "y": 200.3
  },
  "goal_points": [
    {
      "x": 800.7,
      "y": 450.2
    },
    {
      "x": 250.1,
      "y": 750.8
    },
    {
      "x": 600.4,
      "y": 300.6
    },
    {
      "x": 400.9,
      "y": 900.1
    }
  ],
  "scale_pix_to_m": 0.02,
  "k_top_paths": 5,
  "hausdorff_tolerance": 15.0
}
```

## Complete cURL Example

```bash
curl -X POST "http://localhost:8000/plan_path" \
  -H "Content-Type: multipart/form-data" \
  -F 'request_data="{
    \"start_point\": {\"x\": 454, \"y\": 368},
    \"goal_points\": [
      {\"x\": 852, \"y\": 460},
      {\"x\": 234, \"y\": 830},
      {\"x\": 302, \"y\": 990}
    ],
    \"scale_pix_to_m\": 0.05,
    \"k_top_paths\": 3,
    \"hausdorff_tolerance\": 10.0
  }"' \
  -F "segmentation_image=@path/to/segmentation_image.png" \
  -F "real_world_image=@path/to/real_world_image.jpg"
```

## Python Requests Example

```python
import requests
import json

# Prepare request data
request_data = {
    "start_point": {"x": 454, "y": 368},
    "goal_points": [
        {"x": 852, "y": 460},
        {"x": 234, "y": 830},
        {"x": 302, "y": 990}
    ],
    "scale_pix_to_m": 0.05,
    "k_top_paths": 3,
    "hausdorff_tolerance": 10.0
}

# Prepare files
files = {
    'request_data': ('', json.dumps(request_data)),
    'segmentation_image': ('seg.png', open('segmentation_image.png', 'rb'), 'image/png'),
    'real_world_image': ('real.jpg', open('real_world_image.jpg', 'rb'), 'image/jpeg')
}

# Make request
response = requests.post('http://localhost:8000/plan_path', files=files)

# Handle response
if response.status_code == 200:
    result = response.json()
    print(f"Success: {result['success']}")
    print(f"Paths found: {result['total_paths_found']}")
    
    # Access visualization images (base64 encoded)
    for viz_name, base64_img in result['visualization_images'].items():
        print(f"Visualization available: {viz_name}")
        # To decode: 
        # import base64
        # img_bytes = base64.b64decode(base64_img)
        
else:
    print(f"Error: {response.status_code} - {response.text}")
```

## JavaScript/Fetch Example

```javascript
const requestData = {
    start_point: { x: 454, y: 368 },
    goal_points: [
        { x: 852, y: 460 },
        { x: 234, y: 830 },
        { x: 302, y: 990 }
    ],
    scale_pix_to_m: 0.05,
    k_top_paths: 3,
    hausdorff_tolerance: 10.0
};

const formData = new FormData();
formData.append('request_data', JSON.stringify(requestData));
formData.append('segmentation_image', segmentationImageFile);
formData.append('real_world_image', realWorldImageFile);

fetch('http://localhost:8000/plan_path', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log('Success:', data);
    // Handle the response data
})
.catch(error => {
    console.error('Error:', error);
});
```

## Field Descriptions

- **start_point**: Starting pixel coordinates (x, y)
- **goal_points**: Array of destination pixel coordinates
- **scale_pix_to_m**: Conversion factor from pixels to meters (default: 0.05 = 5cm per pixel)
- **k_top_paths**: Number of best paths to return (default: 3)
- **hausdorff_tolerance**: Tolerance for path similarity detection (default: 10.0)

## Image Requirements

1. **segmentation_image**: Binary image where:
   - White pixels (255) = Roads/passable areas
   - Black pixels (0) = Obstacles/impassable areas
   - Formats: PNG, JPG, JPEG

2. **real_world_image** (optional): Real-world satellite/aerial image
   - Used for visualization overlay
   - Should correspond to the same area as segmentation image
   - Formats: PNG, JPG, JPEG

## Response Structure

The API returns a JSON response with:
- `success`: Boolean indicating if path planning succeeded
- `message`: Description of the result
- `total_paths_found`: Number of valid paths found
- `results`: Array of path results with coordinates and metrics
- `visualization_images`: Base64 encoded visualization plots