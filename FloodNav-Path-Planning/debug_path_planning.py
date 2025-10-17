"""
Path Planning Debug Tool
Validates images and coordinates for path planning issues.

Usage:
    python debug_path_planning.py
"""

import cv2
import numpy as np
import json
import sys
from pathlib import Path

def analyze_segmentation_image(image_path):
    """Analyze segmentation image for path planning"""
    print(f"🔍 Analyzing segmentation image: {image_path}")
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return None
    
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ Could not load image: {image_path}")
        return None
    
    height, width = img.shape
    print(f"📐 Image size: {width}x{height}")
    
    # Analyze pixel values
    unique_values = np.unique(img)
    print(f"🎨 Unique pixel values: {unique_values}")
    
    # Count roads vs obstacles
    total_pixels = height * width
    
    if len(unique_values) == 2 and 0 in unique_values and 255 in unique_values:
        # Binary image
        road_pixels = np.sum(img == 255)
        obstacle_pixels = np.sum(img == 0)
        print(f"🛣️  Roads (white/255): {road_pixels} pixels ({road_pixels/total_pixels*100:.1f}%)")
        print(f"🚫 Obstacles (black/0): {obstacle_pixels} pixels ({obstacle_pixels/total_pixels*100:.1f}%)")
    else:
        # Non-binary image
        print(f"⚠️  Image is not binary! Path planning expects:")
        print(f"   • Roads: white pixels (255)")
        print(f"   • Obstacles: black pixels (0)")
        
        # Try to suggest threshold
        if unique_values.max() > 127:
            threshold = 127
            binary_img = (img > threshold).astype(np.uint8) * 255
            road_pixels = np.sum(binary_img == 255)
            print(f"💡 Suggested binary conversion (threshold={threshold}):")
            print(f"   • Roads: {road_pixels} pixels ({road_pixels/total_pixels*100:.1f}%)")
    
    return {
        'shape': (width, height),
        'unique_values': unique_values.tolist(),
        'total_pixels': total_pixels,
        'is_binary': len(unique_values) == 2 and 0 in unique_values and 255 in unique_values
    }

def check_coordinates(coordinates, image_shape):
    """Check if coordinates are valid for the image"""
    width, height = image_shape
    
    print(f"\n🎯 Checking coordinates for image size {width}x{height}:")
    
    # Check start point
    start = coordinates['start_point']
    print(f"📍 Start point: ({start['x']}, {start['y']})")
    
    if 0 <= start['x'] < width and 0 <= start['y'] < height:
        print(f"   ✅ Start point is within bounds")
    else:
        print(f"   ❌ Start point is OUT OF BOUNDS!")
        return False
    
    # Check goal points
    all_valid = True
    for i, goal in enumerate(coordinates['goal_points']):
        print(f"🎯 Goal {i+1}: ({goal['x']}, {goal['y']})")
        
        if 0 <= goal['x'] < width and 0 <= goal['y'] < height:
            print(f"   ✅ Goal {i+1} is within bounds")
        else:
            print(f"   ❌ Goal {i+1} is OUT OF BOUNDS!")
            all_valid = False
    
    return all_valid

def check_coordinate_accessibility(image_path, coordinates):
    """Check if start/goal coordinates are on passable areas"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    
    print(f"\n🚦 Checking coordinate accessibility:")
    
    # Check start point
    start = coordinates['start_point']
    start_pixel = img[start['y'], start['x']]
    print(f"📍 Start point pixel value: {start_pixel}")
    
    if start_pixel == 255:
        print(f"   ✅ Start point is on a road (white)")
    elif start_pixel == 0:
        print(f"   ❌ Start point is on an obstacle (black)!")
        return False
    else:
        print(f"   ⚠️  Start point pixel value is {start_pixel} (expected 0 or 255)")
    
    # Check goal points
    all_accessible = True
    for i, goal in enumerate(coordinates['goal_points']):
        goal_pixel = img[goal['y'], goal['x']]
        print(f"🎯 Goal {i+1} pixel value: {goal_pixel}")
        
        if goal_pixel == 255:
            print(f"   ✅ Goal {i+1} is on a road (white)")
        elif goal_pixel == 0:
            print(f"   ❌ Goal {i+1} is on an obstacle (black)!")
            all_accessible = False
        else:
            print(f"   ⚠️  Goal {i+1} pixel value is {goal_pixel} (expected 0 or 255)")
    
    return all_accessible

def create_debug_visualization(image_path, coordinates, output_path):
    """Create a debug visualization showing coordinates on the image"""
    img = cv2.imread(image_path)
    if img is None:
        return False
    
    # Draw start point (green)
    start = coordinates['start_point']
    cv2.circle(img, (start['x'], start['y']), 8, (0, 255, 0), -1)
    cv2.putText(img, "START", (start['x'] + 10, start['y'] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Draw goal points (red)
    for i, goal in enumerate(coordinates['goal_points']):
        cv2.circle(img, (goal['x'], goal['y']), 6, (0, 0, 255), -1)
        cv2.putText(img, f"G{i+1}", (goal['x'] + 10, goal['y'] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Save debug image
    cv2.imwrite(output_path, img)
    print(f"💾 Debug visualization saved: {output_path}")
    return True

def main():
    print("="*70)
    print("🐛 PATH PLANNING DEBUG TOOL")
    print("="*70)
    
    # Your test data
    segmentation_image = "resource/imagepan5.png"
    coordinates = {
        "start_point": {"x": 44, "y": 161},
        "goal_points": [
            {"x": 255, "y": 358},
            {"x": 121, "y": 420},
            {"x": 222, "y": 30}
        ]
    }
    
    # Analyze segmentation image
    analysis = analyze_segmentation_image(segmentation_image)
    if analysis is None:
        return
    
    # Check coordinate bounds
    coords_valid = check_coordinates(coordinates, analysis['shape'])
    
    # Check coordinate accessibility
    coords_accessible = check_coordinate_accessibility(segmentation_image, coordinates)
    
    # Create debug visualization
    debug_output = "debug_coordinates.png"
    create_debug_visualization(segmentation_image, coordinates, debug_output)
    
    print(f"\n{'='*70}")
    print("📋 DIAGNOSIS SUMMARY")
    print("="*70)
    print(f"✅ Image loaded: {Path(segmentation_image).exists()}")
    print(f"✅ Binary format: {analysis['is_binary']}")
    print(f"✅ Coordinates in bounds: {coords_valid}")
    print(f"✅ Coordinates accessible: {coords_accessible}")
    
    if not analysis['is_binary']:
        print(f"\n💡 RECOMMENDATION: Convert image to binary format")
        print(f"   • Roads should be white (255)")
        print(f"   • Obstacles should be black (0)")
    
    if not coords_accessible:
        print(f"\n💡 RECOMMENDATION: Move start/goal points to white (road) areas")
        print(f"   • Use the coordinate picker tool to select valid points")
    
    print(f"\n🎨 Debug visualization: {debug_output}")
    print("="*70)

if __name__ == "__main__":
    main()