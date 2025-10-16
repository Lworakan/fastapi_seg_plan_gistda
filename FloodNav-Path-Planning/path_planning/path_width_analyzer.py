import numpy as np
import statistics
from scipy.ndimage import distance_transform_edt


class PathWidthAnalyzer:
    """
    Analyzes road widths along planned paths using distance transform and fallback methods.
    """
    
    def __init__(self, debug=True):
        """
        Initialize the PathWidthAnalyzer.
        
        Args:
            debug: Enable debug output
        """
        self.debug = debug
    
    def analyze_grid_format(self, grid, sample_points=10):
        """Analyze and display grid format information."""
        if not self.debug:
            return
            
        print(f"Grid shape: {grid.shape}")
        print(f"Grid dtype: {grid.dtype}")
        print(f"Grid min value: {np.min(grid)}")
        print(f"Grid max value: {np.max(grid)}")
        print(f"Unique values in grid: {np.unique(grid)}")
        
        # Sample some points to understand the format
        if len(np.unique(grid)) <= 10:  # If limited unique values, show all
            unique, counts = np.unique(grid, return_counts=True)
            print("Value distribution:")
            for val, count in zip(unique, counts):
                print(f"  {val}: {count} pixels ({100*count/grid.size:.1f}%)")
    
    def validate_grid_interpretation(self, grid, rx, ry, sample_size=20):
        """
        Validate grid interpretation by checking which format works with the path.
        
        Args:
            grid: The planning grid
            rx, ry: Path coordinates
            sample_size: Number of path points to sample for validation
            
        Returns:
            tuple: (passable_mask, interpretation_description)
        """
        # Sample some path points for validation
        sample_indices = np.linspace(0, len(rx)-1, min(sample_size, len(rx))).astype(int)
        
        unique_vals = sorted(np.unique(grid))
        
        if len(unique_vals) == 2:
            val1, val2 = unique_vals
            
            # Try interpretation 1: val1=passable, val2=obstacle
            mask1 = (grid == val1).astype(np.uint8)
            valid1 = 0
            for i in sample_indices:
                x, y = int(round(rx[i])), int(round(ry[i]))
                if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]:
                    if mask1[y, x] == 1:
                        valid1 += 1
            
            # Try interpretation 2: val2=passable, val1=obstacle  
            mask2 = (grid == val2).astype(np.uint8)
            valid2 = 0
            for i in sample_indices:
                x, y = int(round(rx[i])), int(round(ry[i]))
                if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]:
                    if mask2[y, x] == 1:
                        valid2 += 1
            
            if self.debug:
                print(f"Grid interpretation validation:")
                print(f"  {val1}=passable, {val2}=obstacle: {valid1}/{len(sample_indices)} path points valid")
                print(f"  {val2}=passable, {val1}=obstacle: {valid2}/{len(sample_indices)} path points valid")
            
            if valid1 > valid2:
                return mask1, f"{val1}=passable, {val2}=obstacle"
            else:
                return mask2, f"{val2}=passable, {val1}=obstacle"
        
        else:
            # Multi-value case - try different thresholds
            best_mask = None
            best_description = ""
            best_valid = 0
            
            for threshold in [np.min(grid), np.median(grid), np.mean(grid)]:
                # Try lower values = passable
                mask_low = (grid <= threshold).astype(np.uint8)
                valid_low = 0
                for i in sample_indices:
                    x, y = int(round(rx[i])), int(round(ry[i]))
                    if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]:
                        if mask_low[y, x] == 1:
                            valid_low += 1
                
                # Try higher values = passable
                mask_high = (grid >= threshold).astype(np.uint8)
                valid_high = 0
                for i in sample_indices:
                    x, y = int(round(rx[i])), int(round(ry[i]))
                    if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]:
                        if mask_high[y, x] == 1:
                            valid_high += 1
                
                if valid_low > best_valid:
                    best_valid = valid_low
                    best_mask = mask_low
                    best_description = f"<={threshold:.3f}=passable, >{threshold:.3f}=obstacle"
                
                if valid_high > best_valid:
                    best_valid = valid_high
                    best_mask = mask_high  
                    best_description = f">={threshold:.3f}=passable, <{threshold:.3f}=obstacle"
            
            if self.debug:
                print(f"Best interpretation: {best_description} ({best_valid}/{len(sample_indices)} valid)")
            
            return best_mask, best_description
    
    def calculate_path_widths(self, grid, rx, ry, sample_interval=5):
        """
        Calculate minimum and maximum road widths along a path.
        
        Args:
            grid: The planning grid
            rx, ry: Path coordinates
            sample_interval: Sample every N points along the path
            
        Returns:
            dict: Width statistics including min, max, avg, and individual measurements
        """
        if self.debug:
            print("\n--- DEBUGGING GRID FORMAT ---")
            self.analyze_grid_format(grid)
        
        # Validate grid interpretation using the actual path
        if self.debug:
            print("\n--- VALIDATING GRID INTERPRETATION ---")
        
        passable_mask, interpretation = self.validate_grid_interpretation(grid, rx, ry)
        
        if self.debug:
            print(f"Using interpretation: {interpretation}")
            print(f"Passable area percentage: {100*np.sum(passable_mask)/passable_mask.size:.1f}%")
        
        # Calculate distance transform - distance from each passable point to nearest obstacle
        distance_map = distance_transform_edt(passable_mask)
        
        if self.debug:
            print(f"Distance map - min: {distance_map.min():.2f}, max: {distance_map.max():.2f}")
            print(f"Non-zero distance points: {np.sum(distance_map > 0)}")
        
        # Sample points along the path
        path_indices = range(0, len(rx), sample_interval)
        if len(rx) - 1 not in path_indices:
            path_indices = list(path_indices) + [len(rx) - 1]
        
        widths = []
        valid_samples = 0
        invalid_count = 0
        
        for i in path_indices:
            x, y = int(round(rx[i])), int(round(ry[i]))
            
            # Ensure coordinates are within bounds
            if 0 <= y < distance_map.shape[0] and 0 <= x < distance_map.shape[1]:
                # Check if point is on passable area
                if passable_mask[y, x] == 1:
                    # Distance to nearest obstacle
                    dist_to_obstacle = distance_map[y, x]
                    # Road width is approximately 2 * distance to nearest obstacle
                    road_width = 2.0 * dist_to_obstacle
                    widths.append(road_width)
                    valid_samples += 1
                    
                    # Debug first few points
                    if self.debug and len(widths) <= 3:
                        print(f"Point ({x},{y}): distance={dist_to_obstacle:.2f}, width={road_width:.2f}")
                else:
                    invalid_count += 1
                    if self.debug and invalid_count <= 5:  # Only print first 5 warnings
                        print(f"Warning: Path point ({x},{y}) is on obstacle area!")
                    elif self.debug and invalid_count == 6:
                        print(f"... (suppressing further warnings)")
        
        if self.debug:
            print(f"Valid width samples: {valid_samples}/{len(path_indices)} (invalid: {invalid_count})")
        
        if not widths:
            if self.debug:
                print("No valid width measurements found!")
                print("Attempting fallback width calculation...")
            return self.calculate_width_fallback(grid, rx, ry, sample_interval)
        
        result = {
            'min_width': min(widths),
            'max_width': max(widths),
            'avg_width': statistics.mean(widths),
            'widths': widths,
        }
        
        if self.debug:
            print(f"Width analysis complete: min={result['min_width']:.2f}px, "
                  f"max={result['max_width']:.2f}px, "
                  f"avg={result['avg_width']:.2f}px")
        
        return result
    
    def calculate_width_fallback(self, grid, rx, ry, sample_interval=5):
        """
        Fallback width calculation using perpendicular line scanning.
        
        Args:
            grid: The planning grid
            rx, ry: Path coordinates
            sample_interval: Sample every N points along the path
            
        Returns:
            dict: Width statistics from fallback method
        """
        if self.debug:
            print("Using fallback method: perpendicular line scanning...")
        
        # Determine what value represents passable areas by checking path points
        path_values = []
        for i in range(0, len(rx), max(1, len(rx)//10)):
            x, y = int(round(rx[i])), int(round(ry[i]))
            if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]:
                path_values.append(grid[y, x])
        
        if not path_values:
            return {'min_width': 0, 'max_width': 0, 'avg_width': 0, 'widths': []}
        
        # The most common value along the path should be the passable value
        passable_value = statistics.mode(path_values)
        if self.debug:
            print(f"Detected passable value from path: {passable_value}")
        
        widths = []
        path_indices = range(0, len(rx), sample_interval)
        
        for i in path_indices:
            if i >= len(rx) - 1:
                continue
                
            x, y = int(round(rx[i])), int(round(ry[i]))
            
            # Get direction vector (approximate)
            if i < len(rx) - 1:
                dx = rx[i+1] - rx[i]
                dy = ry[i+1] - ry[i]
            else:
                dx = rx[i] - rx[i-1] 
                dy = ry[i] - ry[i-1]
            
            # Normalize direction
            length = (dx*dx + dy*dy)**0.5
            if length > 0:
                dx, dy = dx/length, dy/length
            else:
                continue
            
            # Perpendicular direction
            perp_x, perp_y = -dy, dx
            
            # Scan in both perpendicular directions to find width
            width = 0
            for direction in [-1, 1]:
                scan_x, scan_y = perp_x * direction, perp_y * direction
                distance = 0
                
                for step in range(1, 100):  # Maximum scan distance
                    check_x = int(round(x + scan_x * step))
                    check_y = int(round(y + scan_y * step))
                    
                    if (check_x < 0 or check_x >= grid.shape[1] or 
                        check_y < 0 or check_y >= grid.shape[0] or
                        grid[check_y, check_x] != passable_value):
                        distance = step
                        break
                
                width += distance
            
            if width > 0:
                widths.append(float(width))
        
        if not widths:
            return {'min_width': 0, 'max_width': 0, 'avg_width': 0, 'widths': []}
        
        result = {
            'min_width': min(widths),
            'max_width': max(widths),  
            'avg_width': statistics.mean(widths),
            'widths': widths,
        }
        
        if self.debug:
            print(f"Fallback width analysis: min={result['min_width']:.2f}px, "
                  f"max={result['max_width']:.2f}px, "
                  f"avg={result['avg_width']:.2f}px")
        
        return result
    
    def analyze_multiple_paths(self, grid, path_results):
        """
        Analyze widths for multiple path planning results.
        
        Args:
            grid: The planning grid
            path_results: List of path planning results with 'path' key containing (rx, ry)
            
        Returns:
            list: Path results with added 'width_stats' key
        """
        if self.debug:
            print("\n" + "="*60)
            print("PATH WIDTH ANALYSIS")
            print("="*60)
        
        for i, res in enumerate(path_results):
            rx, ry = res["path"]
            width_stats = self.calculate_path_widths(grid, rx, ry, sample_interval=5)
            
            # Store width stats in the result
            res["width_stats"] = width_stats
            
            if self.debug:
                # Print width statistics
                perm_str = " -> ".join([f"G{idx+1}" for idx, g in enumerate(res.get("perm", []))])
                print(f"\n[Rank {i+1}] {res.get('algo', 'Unknown')} | Order: {perm_str}")
                print(f"  Distance: {res.get('dist', 0):.2f} pixels | Time: {res.get('time', 0):.4f}s")
                print(f"  Road Width Analysis:")
                print(f"    Min Width: {width_stats['min_width']:.2f} pixels")
                print(f"    Max Width: {width_stats['max_width']:.2f} pixels")
                print(f"    Avg Width: {width_stats['avg_width']:.2f} pixels")
                print(f"    Width Variation: {width_stats['max_width'] - width_stats['min_width']:.2f} pixels")
        
        return path_results
    
    def find_best_width_characteristics(self, path_results):
        """
        Find paths with best width characteristics.
        
        Args:
            path_results: List of path results with width_stats
            
        Returns:
            dict: Analysis of best paths by different width criteria
        """
        if not path_results:
            return {}
        
        # Find the path with minimum bottleneck (largest minimum width)
        best_min_width = max(path_results, 
                           key=lambda r: r.get("width_stats", {}).get("min_width", 0))
        
        # Find the path with most consistent width (smallest variation)
        width_variations = []
        for res in path_results:
            stats = res.get("width_stats", {})
            if stats and 'min_width' in stats and 'max_width' in stats:
                variation = stats['max_width'] - stats['min_width']
                width_variations.append((variation, res))
        
        most_consistent = None
        if width_variations:
            most_consistent = min(width_variations, key=lambda x: x[0])
        
        analysis = {
            'best_min_width': {
                'rank': path_results.index(best_min_width) + 1 if best_min_width in path_results else 0,
                'algorithm': best_min_width.get('algo', 'Unknown'),
                'min_width_px': best_min_width.get('width_stats', {}).get('min_width', 0),
            }
        }
        
        if most_consistent:
            analysis['most_consistent'] = {
                'rank': path_results.index(most_consistent[1]) + 1,
                'algorithm': most_consistent[1].get('algo', 'Unknown'),
                'width_variation_px': most_consistent[0],
            }
        
        if self.debug:
            print("\n" + "="*60)
            print("PATH WIDTH COMPARISON")
            print("="*60)
            
            print(f"Path with largest minimum width (best for wide vehicles):")
            print(f"  Rank: {analysis['best_min_width']['rank']}")
            print(f"  Algorithm: {analysis['best_min_width']['algorithm']}")
            print(f"  Min Width: {analysis['best_min_width']['min_width_px']:.2f} pixels")
            
            if 'most_consistent' in analysis:
                print(f"\nMost consistent width path:")
                print(f"  Rank: {analysis['most_consistent']['rank']}")
                print(f"  Algorithm: {analysis['most_consistent']['algorithm']}")
                print(f"  Width Variation: {analysis['most_consistent']['width_variation_px']:.2f} pixels ({analysis['most_consistent']['width_variation_m']:.2f}m)")
        
        return analysis