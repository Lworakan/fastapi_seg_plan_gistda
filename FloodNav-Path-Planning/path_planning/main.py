"""
Main Path Planning Script
Demonstrates usage of the refactored path planning classes.
"""

import numpy as np
import cv2

# Import our custom classes
from path_width_analyzer import PathWidthAnalyzer
from path_planning_manager import PathPlanningManager, PlannerFactory
from path_visualizer import PathVisualizer


def main():
    """Main function demonstrating the refactored path planning system."""
    
    # --- Configuration ---
    print("="*60)
    print("MULTI-GOAL PATH PLANNING WITH WIDTH ANALYSIS")
    print("="*60)
    
    # File paths
    grid_file = r"C:\fibo\3rd year_1st semester\THEOS-2\Path Planning\resource\map.npy"
    image_file = r"C:\fibo\3rd year_1st semester\THEOS-2\Path Planning\resource\map.png"
    real_image_file = r"C:\fibo\3rd year_1st semester\THEOS-2\Path Planning\resource\satellite.png" 
    
    # Scale and coordinates
    scale_pix_to_m = 0.05  # 5 cm per pixel
    
    # Define problem: start and goals
    point = {'1': (852, 460), '2': (234, 830), '3': (302, 990), '4': (234, 128)}
    start = (454, 368)
    goal_list = [point['1'], point['2'], point['3'], point['4']]
    
    # Algorithm parameters
    k_top_paths = 3
    hausdorff_tolerance = 10.0
    
    # --- Data Loading ---
    print("\n1. Loading data...")
    try:
        grid = np.load(grid_file)
        image = cv2.imread(image_file)
        print(f"   Grid shape: {grid.shape}")
        print(f"   Image shape: {image.shape if image is not None else 'Not loaded'}")
    except Exception as e:
        print(f"   Error loading data: {e}")
        return
    
    # --- Initialize Classes ---
    print("\n2. Initializing planning system...")
    
    # Create planner factory and get available planners
    planners_dict = PlannerFactory.create_planners_dict(grid)
    print(f"   Available planners: {list(planners_dict.keys())}")
    
    if not planners_dict:
        print("   Error: No planners available. Check planner imports.")
        return
    
    # Initialize manager classes
    planner_manager = PathPlanningManager(grid, planners_dict, debug=True)
    width_analyzer = PathWidthAnalyzer(debug=True)
    visualizer = PathVisualizer()

    width_comparison = 0
    
    # Configure parameters
    planner_manager.set_parameters(
        k_top_paths=k_top_paths, 
        hausdorff_tolerance=hausdorff_tolerance
    )
    
    # --- Run Path Planning Analysis ---
    print("\n3. Running complete path planning analysis...")
    
    try:
        # Run complete planning pipeline
        analysis_results = planner_manager.run_complete_analysis(start, goal_list)
        
        # Get the top results
        top_results = analysis_results['top_results']
        
        if not top_results:
            print("   No valid paths found!")
            return
            
        print(f"\n   Found {len(top_results)} unique optimal paths")
        print(f"   Tested {analysis_results['total_combinations_tested']} combinations")
        
    except Exception as e:
        print(f"   Error during path planning: {e}")
        return
    
    # --- Width Analysis ---
    print("\n4. Analyzing road widths...")
    
    try:
        # Analyze path widths
        top_results_with_width = width_analyzer.analyze_multiple_paths(grid, top_results)
        
        # Find best width characteristics
        c = width_analyzer.find_best_width_characteristics(top_results_with_width)
        
        print("\n   Width analysis complete!")
        
    except Exception as e:
        print(f"   Error during width analysis: {e}")
        # Continue without width analysis
        top_results_with_width = top_results
    
    # --- Visualizations ---
    print("\n5. Creating visualizations...")
    
    figures = {}
    
    try:
        # Grid-based path visualization
        print("   Creating grid path plots...")
        fig_grid = visualizer.plot_grid_paths(
            grid, top_results_with_width, start, goal_list, scale_pix_to_m=0.05, show_width_info=True
        )
        figures['grid_paths'] = fig_grid
        
        # Algorithm performance plot
        print("   Creating performance analysis...")
        fig_perf = visualizer.plot_algorithm_performance(analysis_results['algorithm_performance'])
        figures['algorithm_performance'] = fig_perf
        
        # Width analysis plots (if width data available)
        if any(r.get('width_stats', {}).get('min_width', 0) > 0 for r in top_results_with_width):
            print("   Creating width analysis plots...")
            fig_width = visualizer.plot_width_analysis(top_results_with_width)
            figures['width_analysis'] = fig_width
        
        # Real-world overlay (if real image available)
        try:
            print("   Creating real-world overlay...")
            coordinate_mapper = visualizer.create_coordinate_mapper(
                grid.shape, 
                (1024, 1024),  # Adjust based on your real image size
                flip_y=False
            )
            
            fig_real = visualizer.plot_real_world_overlay(
                real_image_file, top_results_with_width, start, goal_list, 0.05,
                coordinate_mapper, grid.shape
            )
            figures['real_world_overlay'] = fig_real
            
        except Exception as e:
            print(f"   Real-world overlay not available: {e}")
        
    except Exception as e:
        print(f"   Error creating visualizations: {e}")
    
    # --- Results Summary ---
    print("\n6. FINAL RESULTS SUMMARY")
    print("="*60)
    
    # Print path summaries with width info
    for rank, res in enumerate(top_results_with_width, 1):
        goal_labels = [f"G{i+1}" for i in range(len(goal_list))]
        perm_order = [goal_labels[goal_list.index(g)] for g in res["perm"]]
        
        print(f"\n[RANK {rank}] {res['algo']} Algorithm")
        print(f"  Route: {' → '.join(perm_order)}")
        print(f"  Distance: {res['dist']:.2f} pixels ({res['dist']*scale_pix_to_m:.2f} meters)")
        print(f"  Execution Time: {res['time']:.4f} seconds")
        
        width_stats = res.get("width_stats", {})
        if width_stats and width_stats.get('min_width', 0) > 0:
            print(f"  Road Width - Min: {width_stats['min_width']:.1f}px ({width_stats['min_width']*scale_pix_to_m:.2f}m)")
            print(f"              Max: {width_stats['max_width']:.1f}px ({width_stats['max_width']*scale_pix_to_m:.2f}m)")
            print(f"              Avg: {width_stats['avg_width']:.1f}px ({width_stats['avg_width']*scale_pix_to_m:.2f}m)")
    
    # Best paths analysis
    best_paths = analysis_results['best_paths_analysis']
    print(f"\n🏆 SHORTEST PATH: {best_paths['shortest_distance']['algo']} "
          f"({best_paths['shortest_distance']['dist']:.2f}px)")
    print(f"⚡ FASTEST EXECUTION: {best_paths['fastest_execution']['algo']} "
          f"({best_paths['fastest_execution']['time']:.4f}s)")
    
    if width_comparison:
        if 'most_consistent' in width_comparison:
            print(f"📏 MOST CONSISTENT WIDTH: Rank {width_comparison['most_consistent']['rank']} "
                  f"({width_comparison['most_consistent']['width_variation_m']:.2f}m variation)")
    
    print(f"\n📊 ALGORITHMS TESTED: {', '.join(best_paths['algorithms_used'])}")
    print(f"🔍 ALGORITHM DIVERSITY: {best_paths['algorithm_diversity']} different algorithms in top results")
    
    # --- Save Results (Optional) ---
    try:
        print("\n7. Saving plots...")
        saved_files = visualizer.save_all_plots(figures, output_dir="./output_plots")
        print(f"   Saved {len(saved_files)} plot files")
    except Exception as e:
        print(f"   Could not save plots: {e}")
    
    # Show all plots
    print("\n8. Displaying visualizations...")
    print("   Close the plot windows to continue...")
    
    import matplotlib.pyplot as plt
    plt.show()
    
    print("\n✅ Path planning analysis complete!")
    return analysis_results, top_results_with_width, figures


def quick_demo():
    """Quick demo with minimal configuration for testing."""
    print("Running quick demo...")
    
    # Create a simple test grid
    test_grid = np.ones((100, 100))
    test_grid[20:80, 20:80] = 0  # Create a passable area
    
    # Simple test case
    start = (10, 50)
    goals = [(90, 30), (90, 70)]
    
    # Mock planner for demo
    class MockPlanner:
        def __init__(self, grid):
            self.grid = grid
            
        def planning_multi_goal(self, start, goals):
            # Simple straight-line path for demo
            rx = [start[0]]
            ry = [start[1]]
            
            for goal in goals:
                rx.append(goal[0])
                ry.append(goal[1])
            
            return rx, ry
    
    planners = {"Mock": lambda: MockPlanner(test_grid)}
    
    # Run simplified analysis
    manager = PathPlanningManager(test_grid, planners, debug=True)
    analyzer = PathWidthAnalyzer(debug=True)
    visualizer = PathVisualizer()
    
    results = manager.run_complete_analysis(start, goals)
    results_with_width = analyzer.analyze_multiple_paths(test_grid, results['top_results'])
    
    fig = visualizer.plot_grid_paths(test_grid, results_with_width, start, goals)
    
    import matplotlib.pyplot as plt
    plt.show()
    
    print("Demo complete!")


if __name__ == "__main__":
    # Choose which version to run
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        quick_demo()
    else:
        main()