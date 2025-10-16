"""
Path Visualizer Module
Handles visualization of path planning results and real-world mapping.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as patches


class PathVisualizer:
    """
    Visualizes path planning results with various display options.
    """
    
    def __init__(self, figsize_per_panel=(6, 7)):
        """
        Initialize the PathVisualizer.
        
        Args:
            figsize_per_panel: Size per panel for multi-panel plots
        """
        self.figsize_per_panel = figsize_per_panel
        self.colormap = plt.get_cmap('tab10')
    
    def plot_grid_paths(self, grid, path_results, start, goal_list, scale_pix_to_m, show_width_info=True):
        """
        Plot paths on the planning grid.
        
        Args:
            grid: Planning grid
            path_results: List of path planning results
            start: Start position
            goal_list: List of goal positions
            show_width_info: Whether to display width information
            
        Returns:
            matplotlib Figure object
        """
        n_panels = len(path_results)
        fig, axes = plt.subplots(1, n_panels, 
                                figsize=(self.figsize_per_panel[0] * n_panels, 
                                        self.figsize_per_panel[1]))
        
        scale_pix_to_m = scale_pix_to_m

        if n_panels == 1:
            axes = [axes]
        
        # Colors for visit order - use more distinct colors
        n_goals = len(goal_list)
        order_colors = ['#FF6B35', '#F7931E', '#FFD23F', '#06FFA5', '#4ECDC4', 
                       '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8'][:n_goals]
        
        # Map goal coordinates to indices for labeling
        goal_idx_map = {tuple(g): i for i, g in enumerate(goal_list)}
        
        for idx, res in enumerate(path_results):
            ax = axes[idx]
            
            # Display grid
            ax.imshow(np.array(grid), cmap='gray', origin='upper',
                     extent=[0, grid.shape[1], grid.shape[0], 0])
            
            # Plot start point - larger and more visible
            ax.scatter([start[0]], [start[1]], marker='o', s=150, c='red', 
                      edgecolors='white', linewidth=2, label='Start', zorder=5)
            
            # Plot path with better visibility
            rx, ry = res["path"]
            ax.plot([int(x) for x in rx], [int(y) for y in ry], 
                   linewidth=2, color='blue', alpha=1.0, zorder=6)
            
            # Plot goals in visit order with colors - much larger and more visible
            for step, g in enumerate(res["perm"], start=1):
                col = order_colors[step-1]
                # Larger goal markers with thick white outline
                ax.scatter([g[0]], [g[1]], marker='X', s=150, color=col,
                          edgecolors='white', linewidth=2, zorder=5)
                
                # Add step number with better visibility
                ax.text(g[0] + 25, g[1] - 25, str(step),
                       fontsize=8, fontweight='bold', color='white',
                       ha='center', va='center', zorder=11,
                       bbox=dict(boxstyle="circle,pad=0.4", fc=col, alpha=0.9, 
                                edgecolor='white', linewidth=2))
            
            # Create info text
            order_labels = [f"G{goal_idx_map[tuple(g)]+1}" for g in res["perm"]]
            info_text = f"{res['algo']}\n"
            info_text += "Order: " + " → ".join(order_labels) + "\n"
            info_text += f"Distance: {res['dist']*scale_pix_to_m:.2f}m"
            
            # Add width information if available and requested
            if show_width_info:
                width_stats = res.get("width_stats", {})
                if width_stats and width_stats.get('min_width', 0) > 0:
                    info_text += f"\nMin Width: {width_stats['min_width']*scale_pix_to_m:.2f}m"
                    info_text += f"\nMax Width: {width_stats['max_width']*scale_pix_to_m:.2f}m"

            # Position info box to avoid covering goals - check goal positions
            goal_positions = [g for g in res["perm"]]
            min_x = min(pos[0] for pos in goal_positions + [start])
            min_y = min(pos[1] for pos in goal_positions + [start])
            
            # Choose position based on where goals are located
            if min_x < grid.shape[1] * 0.3:  # Goals on left side
                text_x, text_ha = 0.98, 'right'  # Put text on right
            else:  # Goals on right or center
                text_x, text_ha = 0.02, 'left'   # Put text on left
                
            if min_y < grid.shape[0] * 0.3:  # Goals near top
                text_y, text_va = 0.02, 'bottom'  # Put text at bottom
            else:  # Goals at bottom or center
                text_y, text_va = 0.98, 'top'     # Put text at top
            
            # Add info box with better positioning
            ax.text(text_x, text_y, info_text,
                   transform=ax.transAxes, va=text_va, ha=text_ha, fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.6", fc='black', alpha=0.85,
                            edgecolor='white', linewidth=1), 
                   color='white', zorder=15)
            
            ax.set_title(f"Rank {idx+1} Path", fontsize=14, pad=20)
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
            ax.grid(True, alpha=0.3)
        
        # Create figure-level legend with larger, more visible markers
        legend_handles = [
            Line2D([], [], marker='o', linestyle='None', markersize=16,
                   color='red', markeredgecolor='white', markeredgewidth=3, 
                   label='Start')
        ]
        legend_handles += [
            Line2D([], [], marker='X', linestyle='None', markersize=16,
                   color=order_colors[i], markeredgecolor='white', markeredgewidth=3,
                   label=f'Goal {i+1}')
            for i in range(n_goals)
        ]
        legend_handles.append(
            Line2D([], [], color='blue', linewidth=4, label='Planned Path')
        )
        
        fig.legend(handles=legend_handles, loc='upper center', 
                   ncol=min(n_goals + 2, 7), frameon=True, fancybox=True, shadow=True,
                   fontsize=12, markerscale=1.2)
        
        plt.tight_layout(rect=[0, 0, 1, 0.90])
        return fig
    
    def plot_algorithm_performance(self, algo_summary):
        """
        Plot algorithm performance comparison.
        
        Args:
            algo_summary: Dictionary with algorithm performance statistics
            
        Returns:
            matplotlib Figure object
        """
        if not algo_summary:
            print("No algorithm summary data to plot")
            return None
        
        algorithms = list(algo_summary.keys())
        avg_times = [algo_summary[algo]['avg_time'] for algo in algorithms]
        min_times = [algo_summary[algo]['min_time'] for algo in algorithms]
        max_times = [algo_summary[algo]['max_time'] for algo in algorithms]
        counts = [algo_summary[algo]['count'] for algo in algorithms]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Execution times comparison
        x = np.arange(len(algorithms))
        width = 0.35
        
        ax1.bar(x - width/2, avg_times, width, label='Average', alpha=0.8, color='skyblue')
        ax1.bar(x + width/2, min_times, width, label='Minimum', alpha=0.8, color='lightgreen')
        
        # Add error bars for max times
        errors = [max_time - avg_time for max_time, avg_time in zip(max_times, avg_times)]
        ax1.errorbar(x - width/2, avg_times, yerr=errors, fmt='none', 
                    color='black', capsize=5, alpha=0.7)
        
        ax1.set_xlabel('Algorithm')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_title('Algorithm Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(algorithms, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Test counts
        bars = ax2.bar(algorithms, counts, color='orange', alpha=0.7)
        ax2.set_xlabel('Algorithm')
        ax2.set_ylabel('Number of Tests')
        ax2.set_title('Tests Performed per Algorithm')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    
    def create_coordinate_mapper(self, grid_shape, real_image_shape, 
                               grid_bounds=None, real_bounds=None,
                               rotation=0, flip_x=False, flip_y=False):
        """
        Create a coordinate transformation function from grid to real image space.
        
        Args:
            grid_shape: (height, width) of the planning grid
            real_image_shape: (height, width) of the real image
            grid_bounds: ((min_x, min_y), (max_x, max_y)) in grid coordinates
            real_bounds: ((min_x, min_y), (max_x, max_y)) in real image coordinates
            rotation: rotation angle in degrees
            flip_x, flip_y: whether to flip coordinates
        
        Returns:
            Function that maps (grid_x, grid_y) to (real_x, real_y)
        """
        if grid_bounds is None:
            grid_bounds = ((0, 0), (grid_shape[1], grid_shape[0]))
        if real_bounds is None:
            real_bounds = ((0, 0), (real_image_shape[1], real_image_shape[0]))
        
        # Calculate scaling factors
        grid_width = grid_bounds[1][0] - grid_bounds[0][0]
        grid_height = grid_bounds[1][1] - grid_bounds[0][1]
        real_width = real_bounds[1][0] - real_bounds[0][0]
        real_height = real_bounds[1][1] - real_bounds[0][1]
        
        def transform_point(grid_x, grid_y):
            # Normalize to [0,1]
            norm_x = (grid_x - grid_bounds[0][0]) / grid_width
            norm_y = (grid_y - grid_bounds[0][1]) / grid_height
            
            # Apply flips
            if flip_x:
                norm_x = 1 - norm_x
            if flip_y:
                norm_y = 1 - norm_y
            
            # Scale to real coordinates
            real_x = real_bounds[0][0] + norm_x * real_width
            real_y = real_bounds[0][1] + norm_y * real_height
            
            # Apply rotation if needed
            if rotation != 0:
                center_x = real_bounds[0][0] + real_width / 2
                center_y = real_bounds[0][1] + real_height / 2
                
                cos_r = np.cos(np.radians(rotation))
                sin_r = np.sin(np.radians(rotation))
                
                rel_x = real_x - center_x
                rel_y = real_y - center_y
                
                real_x = center_x + rel_x * cos_r - rel_y * sin_r
                real_y = center_y + rel_x * sin_r + rel_y * cos_r
            
            return real_x, real_y
        
        return transform_point
    
    def plot_real_world_overlay(self, real_image_path, path_results, start, goal_list, scale_pix_to_m, 
                              coordinate_mapper=None, grid_shape=None):
        """
        Overlay planned paths on real-world aerial image.
        
        Args:
            real_image_path: Path to the real aerial image
            path_results: Path planning results
            start: Start position in grid coordinates
            goal_list: Goal positions in grid coordinates
            coordinate_mapper: Function to transform coordinates
            grid_shape: Shape of planning grid (for default mapper)
            
        Returns:
            matplotlib Figure object
        """
        try:
            # Load real image
            real_image = cv2.imread(real_image_path)
            if real_image is None:
                raise FileNotFoundError(f"Could not load image: {real_image_path}")
            real_image_rgb = cv2.cvtColor(real_image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading real image: {e}")
            return None
        
        # Create default coordinate mapper if not provided
        if coordinate_mapper is None and grid_shape is not None:
            coordinate_mapper = self.create_coordinate_mapper(
                grid_shape, 
                (real_image_rgb.shape[0], real_image_rgb.shape[1]),
                flip_y=True  # Common for image coordinate systems
            )
        elif coordinate_mapper is None:
            print("Warning: No coordinate mapper provided and no grid_shape specified")
            return None
        
        # Create visualization
        n_panels = len(path_results)
        fig, axes = plt.subplots(1, n_panels, 
                                figsize=(self.figsize_per_panel[0] * n_panels, 
                                        self.figsize_per_panel[1]))
        if n_panels == 1:
            axes = [axes]
        
        # Colors for visualization
        n_goals = len(goal_list)
        order_colors = ['#FF6B35', '#F7931E', '#FFD23F', '#06FFA5', '#4ECDC4', 
                       '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8'][:n_goals]
        
        # Map goal coordinates to indices for labeling
        goal_idx_map = {tuple(g): i for i, g in enumerate(goal_list)}
        
        for idx, res in enumerate(path_results):
            ax = axes[idx]
            
            # Display real image
            ax.imshow(real_image_rgb, extent=[0, real_image_rgb.shape[1], 
                                            real_image_rgb.shape[0], 0])
            
            # Transform and plot start point
            start_real = coordinate_mapper(start[0], start[1])
            ax.scatter([start_real[0]], [start_real[1]], 
                      marker='o', s=150, c='red', edgecolors='white', linewidth=2,
                      label='Start', zorder=5)
                        # Transform and plot path
            rx, ry = res["path"]
            path_real_x, path_real_y = [], []
            
            for x, y in zip(rx, ry):
                real_x, real_y = coordinate_mapper(x, y)
                path_real_x.append(real_x)
                path_real_y.append(real_y)
            
            ax.plot(path_real_x, path_real_y, linewidth=2, color='blue', alpha=1.0, zorder=6)
            
            # Transform and plot goal points in visit order
            for step, g in enumerate(res["perm"], start=1):
                goal_real = coordinate_mapper(g[0], g[1])
                col = order_colors[step-1]
                ax.scatter([goal_real[0]], [goal_real[1]], 
                          marker='X', s=150, color=col, edgecolors='white', linewidth=3, zorder=5)
                
                # Add step number
                ax.text(goal_real[0] + 25, goal_real[1] - 25, str(step),
                       fontsize=8, fontweight='bold', color='white',
                       ha='center', va='center', zorder=11,
                       bbox=dict(boxstyle="circle,pad=0.4", fc=col, alpha=0.9, edgecolor='white', linewidth=2))
        
            
            # Add path information
            order_labels = [f"G{goal_idx_map[tuple(g)]+1}" for g in res["perm"]]
            info_text = f"{res['algo']}\n"
            info_text += "Route: " + " → ".join(order_labels) + "\n"
            info_text += f"Distance: {res['dist']:.0f}px\n"
            info_text += f"Time: {res['time']:.3f}s"
            
            # Add width information if available
            width_stats = res.get("width_stats", {})
            if width_stats and width_stats.get('min_width', 0) > 0:
                    info_text += f"\nMin Width: {width_stats['min_width']*scale_pix_to_m:.2f}m"
                    info_text += f"\nMax Width: {width_stats['max_width']*scale_pix_to_m:.2f}m"

            # ax.text(0.02, 0.98, info_text,
            #        transform=ax.transAxes, va='top', ha='left', fontsize=11,
            #        bbox=dict(boxstyle="round,pad=0.5", fc='black', alpha=0.85), 
            #        color='white')
            
            ax.set_title(f"Rank {idx+1} Path - Real World", fontsize=14, pad=20)
            ax.set_xlim(0, real_image_rgb.shape[1])
            ax.set_ylim(real_image_rgb.shape[0], 0)  # Flip Y for image coordinates
            ax.grid(True, alpha=0.3)
        
        # Create legend
        legend_handles = [
            Line2D([], [], marker='o', linestyle='None', markersize=14,
                   color='red', markeredgecolor='white', markeredgewidth=3, label='Start')
        ]
        legend_handles += [
            Line2D([], [], marker='X', linestyle='None', markersize=14,
                   color=order_colors[i], markeredgecolor='white', markeredgewidth=3,
                   label=f'Goal {i+1}')
            for i in range(n_goals)
        ]
        legend_handles.append(
            Line2D([], [], color='blue', linewidth=4, label='Planned Path')
        )
        
        fig.legend(handles=legend_handles, loc='upper center', 
                   ncol=min(n_goals + 2, 7), frameon=True, fancybox=True, shadow=True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        return fig
    
    def plot_width_analysis(self, path_results):
        """
        Create specialized plots for width analysis.
        
        Args:
            path_results: Path results with width statistics
            
        Returns:
            matplotlib Figure object
        """
        # Filter results with valid width data
        valid_results = [r for r in path_results 
                        if r.get('width_stats', {}).get('min_width', 0) > 0]
        
        if not valid_results:
            print("No valid width data to plot")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        algorithms = [r['algo'] for r in valid_results]
        min_widths = [r['width_stats']['min_width'] for r in valid_results]
        max_widths = [r['width_stats']['max_width'] for r in valid_results]
        avg_widths = [r['width_stats']['avg_width'] for r in valid_results]
        variations = [r['width_stats']['max_width'] - r['width_stats']['min_width'] 
                     for r in valid_results]
        
        # Min widths comparison
        axes[0,0].bar(range(len(valid_results)), min_widths, 
                     color='lightcoral', alpha=0.7)
        axes[0,0].set_title('Minimum Road Widths')
        axes[0,0].set_ylabel('Width (pixels)')
        axes[0,0].set_xticks(range(len(valid_results)))
        axes[0,0].set_xticklabels([f"R{i+1}\n{algo}" for i, algo in enumerate(algorithms)], 
                                 rotation=45, ha='right')
        axes[0,0].grid(True, alpha=0.3)
        
        # Max widths comparison
        axes[0,1].bar(range(len(valid_results)), max_widths, 
                     color='lightblue', alpha=0.7)
        axes[0,1].set_title('Maximum Road Widths')
        axes[0,1].set_ylabel('Width (pixels)')
        axes[0,1].set_xticks(range(len(valid_results)))
        axes[0,1].set_xticklabels([f"R{i+1}\n{algo}" for i, algo in enumerate(algorithms)], 
                                 rotation=45, ha='right')
        axes[0,1].grid(True, alpha=0.3)
        
        # Width variation
        axes[1,0].bar(range(len(valid_results)), variations, 
                     color='orange', alpha=0.7)
        axes[1,0].set_title('Road Width Variation (Max - Min)')
        axes[1,0].set_ylabel('Width Variation (pixels)')
        axes[1,0].set_xticks(range(len(valid_results)))
        axes[1,0].set_xticklabels([f"R{i+1}\n{algo}" for i, algo in enumerate(algorithms)], 
                                 rotation=45, ha='right')
        axes[1,0].grid(True, alpha=0.3)
        
        # Scatter plot: min width vs distance
        distances = [r['dist'] for r in valid_results]
        colors = [self.colormap(i % 10) for i in range(len(valid_results))]
        
        scatter = axes[1,1].scatter(distances, min_widths, c=colors, s=100, alpha=0.7)
        axes[1,1].set_xlabel('Path Distance (pixels)')
        axes[1,1].set_ylabel('Minimum Width (pixels)')
        axes[1,1].set_title('Distance vs Minimum Width')
        axes[1,1].grid(True, alpha=0.3)
        
        # Add labels to points
        for i, (dist, width) in enumerate(zip(distances, min_widths)):
            axes[1,1].annotate(f'R{i+1}', (dist, width), 
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=10, alpha=0.8)
        
        plt.tight_layout()
        return fig
    
    def save_all_plots(self, figures_dict, output_dir="./plots", format='png', dpi=300):
        """
        Save all generated plots to files.
        
        Args:
            figures_dict: Dictionary of {"name": figure} pairs
            output_dir: Directory to save plots
            format: File format (png, pdf, svg, etc.)
            dpi: Resolution for raster formats
        """
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = []
        for name, fig in figures_dict.items():
            if fig is not None:
                filename = f"{name}.{format}"
                filepath = os.path.join(output_dir, filename)
                fig.savefig(filepath, format=format, dpi=dpi, bbox_inches='tight')
                saved_files.append(filepath)
                print(f"Saved: {filepath}")
        
        return saved_files