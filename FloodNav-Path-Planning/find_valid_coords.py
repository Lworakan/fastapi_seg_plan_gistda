"""
Find Valid Connected Coordinates
Helps you find coordinates that are all connected in the road network.
"""

import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime

def flood_fill_connectivity(grid, start):
    """
    Find all pixels connected to start point using flood fill.
    
    Args:
        grid: Binary grid (1=road, 0=obstacle)
        start: Starting coordinate (x, y)
    
    Returns:
        Set of connected coordinates
    """
    if grid[start[1], start[0]] == 0:
        return set()
    
    height, width = grid.shape
    visited = set()
    stack = [start]
    
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    
    while stack:
        x, y = stack.pop()
        
        if (x, y) in visited:
            continue
            
        visited.add((x, y))
        
        # Check all 8 directions
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            if (0 <= nx < width and 0 <= ny < height and 
                (nx, ny) not in visited and grid[ny, nx] == 1):
                stack.append((nx, ny))
    
    return visited

def find_connected_components(grid):
    """Find all connected components in the road network."""
    height, width = grid.shape
    visited_global = set()
    components = []
    
    # Find all road pixels
    road_pixels = [(x, y) for y in range(height) for x in range(width) if grid[y, x] == 1]
    
    for x, y in road_pixels:
        if (x, y) not in visited_global:
            # Find connected component
            component = flood_fill_connectivity(grid, (x, y))
            if component:
                components.append(component)
                visited_global.update(component)
    
    return components

def suggest_valid_coordinates(image_path, num_goals=3):
    """
    Suggest valid coordinates that are all connected.
    
    Args:
        image_path: Path to segmentation image
        num_goals: Number of goal points to suggest
    
    Returns:
        Dictionary with suggested coordinates
    """
    print(f"🔍 Analyzing: {image_path}")
    
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert to binary grid (1=road, 0=obstacle)
    grid = (image > 127).astype(np.uint8)
    height, width = grid.shape
    
    print(f"📐 Image size: {width}x{height}")
    print(f"🛣️  Total road pixels: {np.sum(grid)}")
    
    # Find connected components
    print("🔍 Finding connected components...")
    components = find_connected_components(grid)
    
    print(f"🌐 Found {len(components)} connected components:")
    for i, comp in enumerate(components):
        print(f"   Component {i+1}: {len(comp)} pixels")
    
    if not components:
        print("❌ No road pixels found!")
        return None
    
    # Use the largest connected component
    largest_component = max(components, key=len)
    print(f"✅ Using largest component with {len(largest_component)} pixels")
    
    # Convert to list for random selection
    component_pixels = list(largest_component)
    
    if len(component_pixels) < num_goals + 1:
        print(f"❌ Not enough connected pixels for {num_goals + 1} points")
        return None
    
    # Select random points from the component
    import random
    random.seed(42)  # For reproducible results
    selected_pixels = random.sample(component_pixels, num_goals + 1)
    
    start_point = selected_pixels[0]
    goal_points = selected_pixels[1:]
    
    result = {
        "image_path": image_path,
        "image_size": {"width": width, "height": height},
        "connected_components": len(components),
        "largest_component_size": len(largest_component),
        "start_point": {"x": start_point[0], "y": start_point[1]},
        "goal_points": [{"x": x, "y": y} for x, y in goal_points],
        "fastapi_request": {
            "start_point": {"x": start_point[0], "y": start_point[1]},
            "goal_points": [{"x": x, "y": y} for x, y in goal_points],
            "scale_pix_to_m": 0.05,
            "k_top_paths": 3,
            "hausdorff_tolerance": 10.0
        }
    }
    
    return result

def create_visualization(image_path, coordinates, output_path):
    """Create visualization of the suggested coordinates."""
    image = cv2.imread(image_path)
    if image is None:
        return
    
    # Draw start point (green)
    start = coordinates["start_point"]
    cv2.circle(image, (start["x"], start["y"]), 8, (0, 255, 0), -1)
    cv2.putText(image, "START", (start["x"] + 15, start["y"] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw goal points (red)
    for i, goal in enumerate(coordinates["goal_points"]):
        cv2.circle(image, (goal["x"], goal["y"]), 8, (0, 0, 255), -1)
        cv2.putText(image, f"G{i+1}", (goal["x"] + 15, goal["y"] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imwrite(output_path, image)
    print(f"📸 Visualization saved: {output_path}")

def main():
    print("="*70)
    print("🎯 VALID COORDINATE FINDER")
    print("="*70)
    
    image_path = "resource/imagepan5.png"
    
    try:
        # Find valid coordinates
        coordinates = suggest_valid_coordinates(image_path, num_goals=3)
        
        if coordinates:
            print(f"\n{'='*70}")
            print("✅ SUGGESTED VALID COORDINATES")
            print("="*70)
            print(f"📍 Start: ({coordinates['start_point']['x']}, {coordinates['start_point']['y']})")
            for i, goal in enumerate(coordinates['goal_points']):
                print(f"🎯 Goal {i+1}: ({goal['x']}, {goal['y']})")
            
            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"valid_coordinates_{timestamp}.json"
            
            with open(output_file, 'w') as f:
                json.dump(coordinates, f, indent=2)
            
            print(f"\n💾 Coordinates saved: {output_file}")
            
            # Create visualization
            create_visualization(image_path, coordinates, f"valid_coordinates_{timestamp}.png")
            
            print(f"\n{'='*70}")
            print("🚀 READY-TO-USE CURL COMMAND")
            print("="*70)
            
            request_data = json.dumps(coordinates['fastapi_request'])
            
            print(f"""curl -X POST "http://localhost:8000/plan_path" \\
  -F 'request_data='{request_data}' \\
  -F "segmentation_image=@{image_path}" \\
  -F "real_world_image=@resource/imagepan5raw.png"
""")
            
        else:
            print("❌ Could not find valid coordinates")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()