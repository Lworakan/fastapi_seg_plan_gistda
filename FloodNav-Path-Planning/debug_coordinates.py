#!/usr/bin/env python3
"""
Debug Path Planning Issues - Check segmentation and coordinates
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def analyze_segmentation_and_coordinates():
    """Analyze segmentation image and coordinate validity."""
    
    print("🔍 DEBUGGING PATH PLANNING ISSUES")
    print("=" * 50)
    
    # Load segmentation image
    map_path = "/Users/worakanlasudee/Documents/GitHub/fastapi_seg_plan/FloodNav-Path-Planning/resource/map.png"
    
    if not os.path.exists(map_path):
        print(f"❌ Map file not found: {map_path}")
        return
    
    # Load and process image
    image = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
    print(f"📐 Image shape: {image.shape}")
    print(f"📊 Pixel value range: {image.min()} - {image.max()}")
    
    # Convert to binary (same as FastAPI processing)
    binary_grid = (image > 127).astype(np.uint8)
    
    print(f"🔢 Binary grid: {binary_grid.shape}")
    print(f"   Free space (1): {np.sum(binary_grid == 1)} pixels")
    print(f"   Obstacles (0): {np.sum(binary_grid == 0)} pixels")
    print(f"   Free space %: {(np.sum(binary_grid == 1) / binary_grid.size) * 100:.1f}%")
    
    # Test coordinates from your cURL command
    coordinates = {
        "start": (454, 368),
        "goals": [(852, 460), (234, 830), (302, 990), (234, 128)]
    }
    
    print(f"\n📍 COORDINATE ANALYSIS:")
    print("-" * 25)
    
    # Check if coordinates are in bounds
    height, width = binary_grid.shape
    print(f"Grid bounds: 0-{width-1} (width), 0-{height-1} (height)")
    
    # Check start point
    start_x, start_y = coordinates["start"]
    if 0 <= start_x < width and 0 <= start_y < height:
        start_value = binary_grid[start_y, start_x]
        status = "✅ FREE" if start_value == 1 else "❌ BLOCKED"
        print(f"Start ({start_x}, {start_y}): {status} (value: {start_value})")
    else:
        print(f"Start ({start_x}, {start_y}): ❌ OUT OF BOUNDS")
    
    # Check goal points
    for i, (goal_x, goal_y) in enumerate(coordinates["goals"]):
        if 0 <= goal_x < width and 0 <= goal_y < height:
            goal_value = binary_grid[goal_y, goal_x]
            status = "✅ FREE" if goal_value == 1 else "❌ BLOCKED"
            print(f"Goal {i+1} ({goal_x}, {goal_y}): {status} (value: {goal_value})")
        else:
            print(f"Goal {i+1} ({goal_x}, {goal_y}): ❌ OUT OF BOUNDS")
    
    # Create visualization
    print(f"\n🎨 CREATING VISUALIZATION...")
    print("-" * 30)
    
    plt.figure(figsize=(12, 8))
    
    # Show segmentation
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Segmentation Image')
    plt.colorbar()
    
    # Plot coordinates on original
    plt.plot(start_x, start_y, 'ro', markersize=10, label='Start')
    for i, (gx, gy) in enumerate(coordinates["goals"]):
        plt.plot(gx, gy, 'bo', markersize=8, label=f'Goal {i+1}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Show binary version
    plt.subplot(1, 2, 2)
    plt.imshow(binary_grid, cmap='gray')
    plt.title('Binary Grid (API Processing)')
    plt.colorbar()
    
    # Plot coordinates on binary
    plt.plot(start_x, start_y, 'ro', markersize=10, label='Start')
    for i, (gx, gy) in enumerate(coordinates["goals"]):
        plt.plot(gx, gy, 'bo', markersize=8, label=f'Goal {i+1}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('segmentation_analysis.png', dpi=150, bbox_inches='tight')
    print("💾 Saved visualization: segmentation_analysis.png")
    
    # Suggest better coordinates
    print(f"\n💡 SUGGESTED FIXES:")
    print("-" * 20)
    
    # Find some free spaces
    free_pixels = np.where(binary_grid == 1)
    if len(free_pixels[0]) > 0:
        # Sample some good coordinates
        sample_indices = np.random.choice(len(free_pixels[0]), min(10, len(free_pixels[0])), replace=False)
        
        print("✅ Try these coordinates (guaranteed free spaces):")
        print("Start options:")
        for i in range(min(3, len(sample_indices))):
            idx = sample_indices[i]
            y, x = free_pixels[0][idx], free_pixels[1][idx]
            print(f"   ({x}, {y})")
        
        print("Goal options:")
        for i in range(3, min(8, len(sample_indices))):
            idx = sample_indices[i]
            y, x = free_pixels[0][idx], free_pixels[1][idx]
            print(f"   ({x}, {y})")
    
    # Check connectivity (simple flood fill from start)
    if 0 <= start_x < width and 0 <= start_y < height and binary_grid[start_y, start_x] == 1:
        connected = flood_fill_connectivity(binary_grid, start_x, start_y)
        connected_pixels = np.sum(connected)
        total_free = np.sum(binary_grid == 1)
        
        print(f"\n🔗 CONNECTIVITY ANALYSIS:")
        print(f"   Connected from start: {connected_pixels} pixels")
        print(f"   Total free space: {total_free} pixels")
        print(f"   Connectivity: {(connected_pixels/total_free)*100:.1f}%")
        
        # Check if goals are reachable
        reachable_goals = 0
        for i, (gx, gy) in enumerate(coordinates["goals"]):
            if 0 <= gx < width and 0 <= gy < height:
                if connected[gy, gx]:
                    reachable_goals += 1
                    print(f"   Goal {i+1}: ✅ Reachable")
                else:
                    print(f"   Goal {i+1}: ❌ Not reachable from start")
        
        if reachable_goals == 0:
            print(f"\n🚨 PROBLEM: No goals are reachable from start!")
            print(f"   This explains why path planning fails.")

def flood_fill_connectivity(grid, start_x, start_y):
    """Simple flood fill to check connectivity."""
    connected = np.zeros_like(grid)
    if grid[start_y, start_x] == 0:
        return connected
    
    stack = [(start_x, start_y)]
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    while stack:
        x, y = stack.pop()
        if connected[y, x] or grid[y, x] == 0:
            continue
        
        connected[y, x] = 1
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < grid.shape[1] and 0 <= ny < grid.shape[0] and
                not connected[ny, nx] and grid[ny, nx] == 1):
                stack.append((nx, ny))
    
    return connected

if __name__ == "__main__":
    analyze_segmentation_and_coordinates()
    
    print(f"\n🎯 NEXT STEPS:")
    print("1. Check segmentation_analysis.png")
    print("2. Use suggested coordinates if current ones are blocked")
    print("3. Verify your segmentation image has connected paths")
    print("4. Try the working coordinates in a new cURL command")