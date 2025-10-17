"""
Deep Debug Script for Path Planning Algorithms
Tests individual algorithms to find the root cause.
"""

import numpy as np
import cv2
import sys
import os
from pathlib import Path

# Add path planning modules
sys.path.append('path_planning')

print("="*70)
print("🔬 DEEP PATH PLANNING DEBUG")
print("="*70)

# Load the same image and coordinates
image_path = "resource/imagepan5.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print(f"❌ Could not load image: {image_path}")
    exit(1)

# Convert to binary grid (roads=1, obstacles=0)
binary_grid = (image > 127).astype(np.uint8)

print(f"📐 Grid shape: {binary_grid.shape}")
print(f"🛣️  Roads (1): {np.sum(binary_grid == 1)} pixels")
print(f"🚫 Obstacles (0): {np.sum(binary_grid == 0)} pixels")

# Test coordinates
start = (44, 161)
goals = [(255, 358), (121, 420), (222, 30)]

print(f"\n🎯 Testing coordinates:")
print(f"   Start: {start} -> Grid value: {binary_grid[start[1], start[0]]}")
for i, goal in enumerate(goals):
    print(f"   Goal {i+1}: {goal} -> Grid value: {binary_grid[goal[1], goal[0]]}")

# Test individual algorithms
print(f"\n{'='*70}")
print("🧪 TESTING INDIVIDUAL ALGORITHMS")
print("="*70)

try:
    from A_star import AStar
    
    print("\n🔍 Testing A* Algorithm...")
    astar = AStar(binary_grid)
    
    # Test simple path from start to first goal
    test_goal = goals[0]
    print(f"   Finding path from {start} to {test_goal}")
    
    try:
        path = astar.find_path(start, test_goal)
        if path:
            print(f"   ✅ A* found path with {len(path)} points")
            print(f"   First few points: {path[:5] if len(path) > 5 else path}")
        else:
            print("   ❌ A* returned empty path")
    except Exception as e:
        print(f"   ❌ A* failed: {e}")
        import traceback
        traceback.print_exc()

except ImportError as e:
    print(f"❌ Could not import A*: {e}")

try:
    from breadth_first import BreadthFirstSearch
    
    print("\n🔍 Testing Breadth-First Search...")
    bfs = BreadthFirstSearch(binary_grid)
    
    try:
        path = bfs.find_path(start, goals[0])
        if path:
            print(f"   ✅ BFS found path with {len(path)} points")
        else:
            print("   ❌ BFS returned empty path")
    except Exception as e:
        print(f"   ❌ BFS failed: {e}")

except ImportError as e:
    print(f"❌ Could not import BFS: {e}")

try:
    from best_first import BestFirstSearch
    
    print("\n🔍 Testing Best-First Search...")
    best_first = BestFirstSearch(binary_grid)
    
    try:
        path = best_first.find_path(start, goals[0])
        if path:
            print(f"   ✅ Best-First found path with {len(path)} points")
        else:
            print("   ❌ Best-First returned empty path")
    except Exception as e:
        print(f"   ❌ Best-First failed: {e}")

except ImportError as e:
    print(f"❌ Could not import Best-First: {e}")

# Test PathPlanningManager
print(f"\n{'='*70}")
print("🧪 TESTING PATH PLANNING MANAGER")
print("="*70)

try:
    from path_planning_manager import PathPlanningManager, PlannerFactory
    
    print("🔍 Creating planners...")
    planners_dict = PlannerFactory.create_planners_dict(binary_grid)
    print(f"   Available planners: {list(planners_dict.keys())}")
    
    manager = PathPlanningManager(binary_grid, planners_dict, debug=True)
    
    print(f"\n🔍 Running complete analysis...")
    print(f"   Start: {start}")
    print(f"   Goals: {goals}")
    
    try:
        results = manager.run_complete_analysis(start, goals)
        print(f"   ✅ Analysis completed")
        print(f"   Result keys: {list(results.keys())}")
        print(f"   Top results: {len(results.get('top_results', []))}")
        print(f"   All results: {len(results.get('all_results', []))}")
        
        if results.get('all_results'):
            print(f"   First result: {results['all_results'][0]}")
        
    except Exception as e:
        print(f"   ❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

except ImportError as e:
    print(f"❌ Could not import PathPlanningManager: {e}")
    import traceback
    traceback.print_exc()

print(f"\n{'='*70}")
print("🔍 MANUAL CONNECTIVITY TEST")
print("="*70)

# Simple flood fill to check connectivity
def flood_fill_connectivity(grid, start_pos):
    """Check which areas are reachable from start position"""
    visited = np.zeros_like(grid, dtype=bool)
    stack = [start_pos]
    reachable_count = 0
    
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    while stack:
        x, y = stack.pop()
        
        if visited[y, x]:
            continue
            
        visited[y, x] = True
        reachable_count += 1
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            if (0 <= nx < grid.shape[1] and 
                0 <= ny < grid.shape[0] and 
                not visited[ny, nx] and 
                grid[ny, nx] == 1):  # Only roads
                stack.append((nx, ny))
    
    return visited, reachable_count

print(f"🔍 Testing connectivity from start point {start}...")
visited, reachable = flood_fill_connectivity(binary_grid, start)

print(f"   Reachable road pixels: {reachable}")
print(f"   Total road pixels: {np.sum(binary_grid == 1)}")

# Check if goals are reachable
for i, goal in enumerate(goals):
    if visited[goal[1], goal[0]]:
        print(f"   ✅ Goal {i+1} {goal} is reachable")
    else:
        print(f"   ❌ Goal {i+1} {goal} is NOT reachable")

print(f"\n{'='*70}")
print("📋 SUMMARY")
print("="*70)
print("If all goals show as reachable, the issue is in the algorithm implementation.")
print("If goals are not reachable, the road network is disconnected.")
print("="*70)