"""
Test client for FastAPI Path Planning Server
Demonstrates how to use the path planning API endpoints.
"""

import requests
import json
import base64
import numpy as np
import cv2
from PIL import Image
import io
import matplotlib.pyplot as plt

class PathPlanningClient:
    """Client class for interacting with the Path Planning API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def health_check(self):
        """Check if the API server is running."""
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.json()
        except Exception as e:
            print(f"Error connecting to API: {e}")
            return None
    
    def get_algorithms(self):
        """Get list of available path planning algorithms."""
        try:
            response = requests.get(f"{self.base_url}/algorithms")
            return response.json()
        except Exception as e:
            print(f"Error getting algorithms: {e}")
            return None
    
    def validate_coordinates(self, coordinates, segmentation_image_path):
        """Validate coordinates against segmentation image."""
        try:
            # Prepare coordinate data
            coord_data = [{"x": x, "y": y} for x, y in coordinates]
            
            # Open segmentation image
            with open(segmentation_image_path, 'rb') as f:
                files = {'segmentation_image': f}
                response = requests.post(
                    f"{self.base_url}/validate_coordinates",
                    json=coord_data,
                    files=files
                )
            
            return response.json()
        except Exception as e:
            print(f"Error validating coordinates: {e}")
            return None
    
    def plan_path(self, start_point, goal_points, segmentation_image_path, 
                  real_world_image_path=None, scale_pix_to_m=0.05, 
                  k_top_paths=3, hausdorff_tolerance=10.0):
        """
        Execute path planning.
        
        Args:
            start_point: Tuple (x, y) for start coordinates
            goal_points: List of tuples [(x1, y1), (x2, y2), ...] for goal coordinates
            segmentation_image_path: Path to binary segmentation image
            real_world_image_path: Optional path to real-world image
            scale_pix_to_m: Scale factor from pixels to meters
            k_top_paths: Number of top paths to return
            hausdorff_tolerance: Hausdorff distance tolerance
        
        Returns:
            API response with path planning results
        """
        try:
            # Prepare request data
            request_data = {
                "start_point": {"x": start_point[0], "y": start_point[1]},
                "goal_points": [{"x": x, "y": y} for x, y in goal_points],
                "scale_pix_to_m": scale_pix_to_m,
                "k_top_paths": k_top_paths,
                "hausdorff_tolerance": hausdorff_tolerance
            }
            
            # Prepare files
            files = {}
            with open(segmentation_image_path, 'rb') as f:
                files['segmentation_image'] = f
                
                if real_world_image_path:
                    with open(real_world_image_path, 'rb') as rf:
                        files['real_world_image'] = rf
                        
                        # Send request
                        response = requests.post(
                            f"{self.base_url}/plan_path",
                            data={'request_data': json.dumps(request_data)},
                            files=files
                        )
                else:
                    # Send request without real world image
                    response = requests.post(
                        f"{self.base_url}/plan_path",
                        data={'request_data': json.dumps(request_data)},
                        files=files
                    )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"API Error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"Error during path planning: {e}")
            return None
    
    def save_visualizations(self, response_data, output_dir="./output"):
        """Save visualization images from API response to files."""
        import os
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if 'visualization_images' in response_data:
            for name, base64_image in response_data['visualization_images'].items():
                if base64_image:
                    try:
                        # Decode base64 image
                        image_data = base64.b64decode(base64_image)
                        
                        # Save to file
                        output_path = os.path.join(output_dir, f"{name}.png")
                        with open(output_path, 'wb') as f:
                            f.write(image_data)
                        
                        print(f"Saved visualization: {output_path}")
                    except Exception as e:
                        print(f"Error saving {name}: {e}")

def create_sample_segmentation_image(width=200, height=200, save_path="sample_segmentation.png"):
    """Create a sample binary segmentation image for testing."""
    # Create a binary image with roads (white) and obstacles (black)
    image = np.zeros((height, width), dtype=np.uint8)
    
    # Add some road paths
    # Horizontal road
    image[height//2-10:height//2+10, :] = 255
    
    # Vertical road
    image[:, width//2-10:width//2+10] = 255
    
    # Diagonal road
    for i in range(min(height, width)):
        if i-5 >= 0 and i+5 < height and i-5 >= 0 and i+5 < width:
            image[i-5:i+5, i-5:i+5] = 255
    
    # Add some obstacles (keep as black/0)
    cv2.rectangle(image, (30, 30), (70, 70), 0, -1)
    cv2.rectangle(image, (130, 130), (170, 170), 0, -1)
    
    # Save image
    cv2.imwrite(save_path, image)
    print(f"Created sample segmentation image: {save_path}")
    return save_path

def create_sample_real_world_image(width=200, height=200, save_path="sample_satellite.png"):
    """Create a sample real-world satellite-like image for testing."""
    # Create a colorful satellite-like image
    image = np.random.randint(50, 150, (height, width, 3), dtype=np.uint8)
    
    # Add some green areas (vegetation)
    image[50:100, 50:100] = [34, 139, 34]  # Forest green
    image[120:170, 120:170] = [34, 139, 34]
    
    # Add some blue areas (water)
    image[20:40, 150:190] = [30, 144, 255]  # Dodger blue
    
    # Add some brown areas (buildings/urban)
    image[150:190, 20:60] = [139, 69, 19]  # Saddle brown
    
    # Save image
    cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f"Created sample real-world image: {save_path}")
    return save_path

def main():
    """Demo function showing how to use the Path Planning API."""
    print("FastAPI Path Planning Client Demo")
    print("=" * 50)
    
    # Initialize client
    client = PathPlanningClient()
    
    # 1. Health check
    print("\n1. Checking API health...")
    health = client.health_check()
    if health:
        print(f"   API Status: {health}")
    else:
        print("   API is not accessible. Make sure the server is running.")
        return
    
    # 2. Get available algorithms
    print("\n2. Getting available algorithms...")
    algorithms = client.get_algorithms()
    if algorithms:
        print(f"   Available algorithms: {algorithms['algorithms']}")
    
    # 3. Create sample images for testing
    print("\n3. Creating sample images...")
    seg_image_path = create_sample_segmentation_image()
    real_image_path = create_sample_real_world_image()
    
    # 4. Define test coordinates
    start_point = (50, 50)
    goal_points = [(150, 150), (50, 150), (150, 50)]
    
    print(f"\n4. Test coordinates:")
    print(f"   Start: {start_point}")
    print(f"   Goals: {goal_points}")
    
    # 5. Validate coordinates
    print("\n5. Validating coordinates...")
    validation = client.validate_coordinates(
        [start_point] + goal_points, 
        seg_image_path
    )
    if validation:
        for result in validation['validation_results']:
            status = "✓" if result['is_valid'] else "✗"
            print(f"   {status} ({result['x']}, {result['y']}): {result['message']}")
    
    # 6. Run path planning
    print("\n6. Running path planning...")
    result = client.plan_path(
        start_point=start_point,
        goal_points=goal_points,
        segmentation_image_path=seg_image_path,
        real_world_image_path=real_image_path,
        scale_pix_to_m=0.05,
        k_top_paths=3,
        hausdorff_tolerance=10.0
    )
    
    if result and result['success']:
        print(f"   ✓ Success! Found {result['total_paths_found']} paths")
        print(f"   ✓ Tested {result['total_combinations_tested']} combinations")
        
        # Display path results
        for i, path_result in enumerate(result['results']):
            print(f"\n   Path {i+1}:")
            print(f"     Algorithm: {path_result['algorithm']}")
            print(f"     Distance: {path_result['distance_pixels']:.2f} pixels ({path_result['distance_meters']:.2f} meters)")
            print(f"     Coordinates: {len(path_result['path_coordinates'])} points")
            
            if path_result['width_stats']:
                width_stats = path_result['width_stats']
                print(f"     Width - Min: {width_stats.get('min_width', 'N/A')}, Max: {width_stats.get('max_width', 'N/A')}")
        
        # 7. Save visualizations
        print("\n7. Saving visualizations...")
        client.save_visualizations(result, "./path_planning_output")
        
        print("\n✅ Demo completed successfully!")
        print("Check the './path_planning_output' folder for visualization images.")
        
    else:
        print("   ✗ Path planning failed")
        if result:
            print(f"   Error: {result.get('message', 'Unknown error')}")

if __name__ == "__main__":
    main()