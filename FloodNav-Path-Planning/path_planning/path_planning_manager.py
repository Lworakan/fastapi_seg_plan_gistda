import numpy as np
import time
import itertools
import statistics
from collections import defaultdict


class PathPlanningManager:
    """
    Manages multi-goal path planning using different algorithms and optimization strategies.
    """
    
    def __init__(self, grid, planners_dict, debug=True):
        """
        Initialize the Path Planning Manager.
        
        Args:
            grid: The planning grid (numpy array)
            planners_dict: Dictionary of planner constructors {"name": lambda: PlannerClass(grid)}
            debug: Enable debug output
        """
        self.grid = grid
        self.planners_dict = planners_dict
        self.debug = debug
        
        # Configuration parameters
        self.k_top_paths = 3  # Number of top unique paths to keep
        self.hausdorff_tolerance = 10.0  # Tolerance for geometric similarity
        self.downsample_points = 80  # Points for geometry comparison
        
        # Results storage
        self.all_results = []
        self.top_results = []
        self.algo_summary = {}
    
    def set_parameters(self, k_top_paths=3, hausdorff_tolerance=10.0, downsample_points=80):
        """Set algorithm parameters."""
        self.k_top_paths = k_top_paths
        self.hausdorff_tolerance = hausdorff_tolerance
        self.downsample_points = downsample_points
    
    def _downsample_path(self, rx, ry, n=None):
        """
        Downsample path to fixed number of points for geometric comparison.
        
        Args:
            rx, ry: Path coordinates
            n: Number of points to sample (defaults to self.downsample_points)
            
        Returns:
            numpy array of sampled points
        """
        if n is None:
            n = self.downsample_points
            
        pts = np.column_stack((rx, ry))
        if len(pts) <= n:
            return pts.astype(float)
        idx = np.linspace(0, len(pts) - 1, n).astype(int)
        return pts[idx].astype(float)
    
    def _hausdorff_np(self, pts1, pts2):
        """
        Calculate symmetric Hausdorff distance between two point sets.
        
        Args:
            pts1, pts2: Point sets as numpy arrays
            
        Returns:
            float: Hausdorff distance
        """
        diff = pts1[:, None, :] - pts2[None, :, :]
        d = np.sqrt(np.sum(diff * diff, axis=2))
        d1 = np.max(np.min(d, axis=1))  # pts1 -> pts2
        d2 = np.max(np.min(d, axis=0))  # pts2 -> pts1
        return max(d1, d2)
    
    def _calculate_fallback_distance(self, rx, ry):
        """Calculate path distance if planner doesn't provide it."""
        total_distance = 0.0
        for i in range(1, len(rx)):
            dx = rx[i] - rx[i-1]
            dy = ry[i] - ry[i-1]
            total_distance += (dx*dx + dy*dy) ** 0.5
        return total_distance
    
    def plan_all_permutations(self, start, goal_list):
        """
        Plan paths for all permutations of goals using all available algorithms.
        
        Args:
            start: Start position (x, y)
            goal_list: List of goal positions [(x1, y1), (x2, y2), ...]
            
        Returns:
            list: All successful path planning results
        """
        possibilities = list(itertools.permutations(goal_list))
        self.all_results = []
        
        if self.debug:
            print(f"Planning {len(possibilities)} permutations with {len(self.planners_dict)} algorithms...")
            print(f"Total combinations: {len(possibilities) * len(self.planners_dict)}")
        
        for perm_idx, perm in enumerate(possibilities):
            for algo_name, make_planner in self.planners_dict.items():
                planner = make_planner()
                
                if self.debug:
                    print(f"🔍 Trying {algo_name} for permutation {perm_idx}: {perm}")
                
                t0 = time.time()
                try:
                    out = planner.planning_multi_goal(start, perm)
                    if self.debug:
                        print(f"  ✅ {algo_name} returned: {type(out)}, length: {len(out) if out else 0}")
                except Exception as e:
                    if self.debug:
                        print(f"  ❌ Error in {algo_name} for permutation {perm_idx}: {e}")
                    continue
                t1 = time.time()
                
                if out is None:
                    continue
                    
                # Handle different planner output formats
                if len(out) == 3:
                    rx, ry, total_distance = out
                elif len(out) == 2:
                    rx, ry = out
                    if hasattr(planner, "calculate_path_distance"):
                        total_distance = planner.calculate_path_distance(rx, ry)
                    else:
                        total_distance = self._calculate_fallback_distance(rx, ry)
                else:
                    continue
                
                # Validate path
                if rx is None or ry is None or len(rx) == 0 or len(ry) == 0:
                    continue
                
                self.all_results.append({
                    "algo": algo_name,
                    "dist": float(total_distance),
                    "time": float(t1 - t0),
                    "perm": perm,
                    "path": (rx, ry),
                    "perm_idx": perm_idx
                })
        
        if self.debug:
            print(f"Generated {len(self.all_results)} valid paths")
        
        return self.all_results
    
    def analyze_algorithm_performance(self):
        """Analyze and summarize algorithm performance."""
        algo_times = defaultdict(list)
        for r in self.all_results:
            algo_times[r["algo"]].append(r["time"])
        
        self.algo_summary = {}
        for algo, times in algo_times.items():
            self.algo_summary[algo] = {
                "count": len(times),
                "avg_time": statistics.mean(times),
                "min_time": min(times),
                "max_time": max(times),
                "std_time": statistics.stdev(times) if len(times) > 1 else 0
            }
        
        if self.debug:
            print("\n" + "="*60)
            print("ALGORITHM PERFORMANCE SUMMARY")
            print("="*60)
            for algo, stats in self.algo_summary.items():
                print(f"{algo}: ran {stats['count']} times")
                print(f"  Avg: {stats['avg_time']:.6f}s | Min: {stats['min_time']:.6f}s | Max: {stats['max_time']:.6f}s")
                print(f"  Std: {stats['std_time']:.6f}s\n")
        
        return self.algo_summary
    
    def filter_unique_paths(self):
        """
        Filter to keep only unique top-K paths by permutation and geometry.
        
        Returns:
            list: Filtered unique top paths
        """
        if not self.all_results:
            if self.debug:
                print("No results to filter")
            return []
        
        # Sort all candidates by distance
        all_sorted = sorted(self.all_results, key=lambda r: r["dist"])
        
        # Pick first K that are unique by permutation AND geometry
        unique_results, seen_perms = [], set()
        
        for r in all_sorted:
            if r["perm"] in seen_perms:
                continue
            
            pts_r = self._downsample_path(r["path"][0], r["path"][1])
            
            same_geom = False
            for u in unique_results:
                pts_u = self._downsample_path(u["path"][0], u["path"][1])
                if self._hausdorff_np(pts_r, pts_u) <= self.hausdorff_tolerance:
                    same_geom = True
                    break
            
            if same_geom:
                continue
            
            unique_results.append(r)
            seen_perms.add(r["perm"])
            
            if len(unique_results) == self.k_top_paths:
                break
        
        self.top_results = unique_results
        
        if self.debug:
            print(f"\nFiltered to {len(self.top_results)} unique paths from {len(all_sorted)} total results")
        
        if not self.top_results:
            raise RuntimeError(f"No unique valid paths found. Try increasing hausdorff_tolerance ({self.hausdorff_tolerance}) or checking planners.")
        
        return self.top_results
    
    def print_results_summary(self, goal_list):
        """Print a summary of the top results."""
        if not self.top_results:
            print("No results to summarize")
            return
        
        # Create mapping for goal labels
        goal_idx_map = {tuple(g): i for i, g in enumerate(goal_list)}
        
        print(f"\nTop {len(self.top_results)} UNIQUE Paths (shortest total distance first):")
        print("="*80)
        
        for rank, res in enumerate(self.top_results, 1):
            perm_labels = [f"G{goal_idx_map[tuple(g)]+1}" for g in res["perm"]]
            perm_str = " -> ".join([f"{label}:{g}" for label, g in zip(perm_labels, res["perm"])])
            
            print(f"[{rank}] Permutation: {perm_str}")
            print(f"    Algorithm: {res['algo']}")
            print(f"    Total Distance: {res['dist']:.4f} pixels")
            print(f"    Execution Time: {res['time']:.4f} seconds")
            
            # Add width info if available
            width_stats = res.get("width_stats", {})
            if width_stats and width_stats.get('min_width', 0) > 0:
                print(f"    Min Road Width: {width_stats['min_width']:.2f} pixels")
                print(f"    Max Road Width: {width_stats['max_width']:.2f} pixels")
            
            print()
    
    def get_best_paths_by_criteria(self):
        """
        Get best paths according to different criteria.
        
        Returns:
            dict: Best paths by various criteria
        """
        if not self.top_results:
            return {}
        
        analysis = {}
        
        # Shortest distance
        analysis['shortest_distance'] = min(self.top_results, key=lambda r: r['dist'])
        
        # Fastest execution
        analysis['fastest_execution'] = min(self.top_results, key=lambda r: r['time'])
        
        # Most efficient algorithm (best time per distance unit)
        efficiency_scores = []
        for r in self.top_results:
            efficiency = r['time'] / r['dist']  # Lower is better
            efficiency_scores.append((efficiency, r))
        
        if efficiency_scores:
            analysis['most_efficient'] = min(efficiency_scores, key=lambda x: x[0])[1]
        
        # Algorithm diversity
        algos_used = set(r['algo'] for r in self.top_results)
        analysis['algorithms_used'] = list(algos_used)
        analysis['algorithm_diversity'] = len(algos_used)
        
        return analysis
    
    def run_complete_analysis(self, start, goal_list):
        """
        Run complete path planning analysis pipeline.
        
        Args:
            start: Start position
            goal_list: List of goal positions
            
        Returns:
            dict: Complete analysis results
        """
        # Step 1: Plan all permutations
        all_results = self.plan_all_permutations(start, goal_list)
        
        if not all_results:
            raise RuntimeError("No valid paths found with any algorithm")
        
        # Step 2: Analyze algorithm performance
        algo_perf = self.analyze_algorithm_performance()
        
        # Step 3: Filter to unique top paths
        top_paths = self.filter_unique_paths()
        
        # Step 4: Print summary
        if self.debug:
            self.print_results_summary(goal_list)
        
        # Step 5: Get best paths by criteria
        best_paths = self.get_best_paths_by_criteria()
        
        return {
            'all_results': all_results,
            'top_results': top_paths,
            'algorithm_performance': algo_perf,
            'best_paths_analysis': best_paths,
            'total_combinations_tested': len(all_results),
            'unique_paths_found': len(top_paths)
        }


class PlannerFactory:
    """Factory for creating planner instances."""
    
    @staticmethod
    def create_planners_dict(grid, planners_to_include=None):
        """
        Create a dictionary of available planners.
        
        Args:
            grid: Planning grid
            planners_to_include: List of planner names to include (None for all)
            
        Returns:
            dict: Dictionary of planner constructors
        """
        # Import planners dynamically to avoid import errors if modules don't exist
        available_planners = {}
        
        try:
            from path_planning.A_star import AStarPlanner
            available_planners["A*"] = lambda: AStarPlanner(grid)
        except ImportError:
            try:
                from .A_star import AStarPlanner
                available_planners["A*"] = lambda: AStarPlanner(grid)
            except ImportError:
                print("A* planner not available")
        
        try:
            from path_planning.breadth_first import BreadthFirstSearchPlanner
            available_planners["Breadth-First"] = lambda: BreadthFirstSearchPlanner(grid)
        except ImportError:
            try:
                from .breadth_first import BreadthFirstSearchPlanner
                available_planners["Breadth-First"] = lambda: BreadthFirstSearchPlanner(grid)
            except ImportError:
                print("Breadth-First planner not available")
        
        try:
            from path_planning.best_first import BestFirstSearchPlanner
            available_planners["Best-First"] = lambda: BestFirstSearchPlanner(grid)
        except ImportError:
            try:
                from .best_first import BestFirstSearchPlanner
                available_planners["Best-First"] = lambda: BestFirstSearchPlanner(grid)
            except ImportError:
                print("Best-First planner not available")
        
        # Filter planners if specific ones requested
        if planners_to_include:
            filtered_planners = {}
            for name in planners_to_include:
                if name in available_planners:
                    filtered_planners[name] = available_planners[name]
                else:
                    print(f"Warning: Planner '{name}' not available")
            available_planners = filtered_planners
        
        return available_planners