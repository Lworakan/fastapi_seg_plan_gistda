"""
FastAPI Path Planning Server
Provides API endpoints for multi-goal path planning with real-world mapping and visualization.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image
import json
import traceback
import tempfile
import os

# Import path planning classes
from path_planning.path_width_analyzer import PathWidthAnalyzer
from path_planning.path_planning_manager import PathPlanningManager, PlannerFactory
from path_planning.path_visualizer import PathVisualizer

# Initialize FastAPI app
app = FastAPI(
    title="Path Planning API",
    description="Multi-goal path planning with real-world mapping and visualization",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Pydantic models for API requests and responses
class CoordinatePoint(BaseModel):
    x: float
    y: float

class PathPlanningRequest(BaseModel):
    start_point: CoordinatePoint
    goal_points: List[CoordinatePoint]
    scale_pix_to_m: float = 0.05  # Default: 5 cm per pixel
    k_top_paths: int = 3
    hausdorff_tolerance: float = 10.0

class PathResult(BaseModel):
    path_id: int
    algorithm: str
    distance_pixels: float
    distance_meters: float
    path_coordinates: List[Tuple[float, float]]
    width_stats: Optional[Dict[str, Any]] = None  # Changed from Dict[str, float] to allow lists

class PathPlanningResponse(BaseModel):
    success: bool
    message: str
    total_paths_found: int
    total_combinations_tested: int
    results: List[PathResult]
    algorithm_performance: Dict[str, Any]
    visualization_images: Dict[str, str]  # Base64 encoded images
    results_directory: Optional[str] = None  # Path to saved results folder

class PathPlanningService:
    """Service class to handle path planning operations."""
    
    def __init__(self):
        self.planner_manager = None
        self.width_analyzer = PathWidthAnalyzer(debug=True)  # Enable debug
        self.visualizer = PathVisualizer()
        self.current_grid = None
        self.current_real_image = None
        
    def load_segmentation_image(self, image_data: bytes) -> np.ndarray:
        """Load and process segmentation image (binary road mask)."""
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                raise ValueError("Failed to decode segmentation image")
            
            # Ensure binary image (0 and 1)
            # Assuming white pixels (255) are roads (1) and black pixels (0) are obstacles (0)
            binary_grid = (image > 127).astype(np.uint8)
            
            return binary_grid
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing segmentation image: {str(e)}")
    
    def load_real_world_image(self, image_data: bytes) -> np.ndarray:
        """Load real-world image for visualization overlay."""
        try:
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Failed to decode real-world image")
            
            # Convert BGR to RGB for matplotlib
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing real-world image: {str(e)}")
    
    def initialize_planners(self, grid: np.ndarray):
        """Initialize path planning algorithms."""
        try:
            planners_dict = PlannerFactory.create_planners_dict(grid)
            if not planners_dict:
                raise ValueError("No planners available")
            
            self.planner_manager = PathPlanningManager(grid, planners_dict, debug=True)  # Enable debug
            return True
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error initializing planners: {str(e)}")
    
    def run_path_planning(self, request: PathPlanningRequest) -> Dict[str, Any]:
        """Execute path planning algorithm."""
        try:
            if self.planner_manager is None or self.current_grid is None:
                raise ValueError("Path planning system not properly initialized")
            
            # Convert coordinate points to tuples with INTEGER coordinates
            # Path planning algorithms require integer indices for numpy arrays
            start = (int(request.start_point.x), int(request.start_point.y))
            goals = [(int(point.x), int(point.y)) for point in request.goal_points]
            
            print(f"🔍 DEBUG: Start: {start}, Goals: {goals}")
            
            # Configure parameters
            self.planner_manager.set_parameters(
                k_top_paths=request.k_top_paths,
                hausdorff_tolerance=request.hausdorff_tolerance
            )
            
            print(f"🔍 DEBUG: Running analysis with {len(goals)} goals...")
            
            # Run path planning
            try:
                analysis_results = self.planner_manager.run_complete_analysis(start, goals)
            except Exception as e:
                print(f"🔍 DEBUG: ERROR in run_complete_analysis: {type(e).__name__}: {str(e)}")
                import traceback
                traceback.print_exc()
                raise
            
            print(f"🔍 DEBUG: Analysis results: {analysis_results.keys()}")
            print(f"🔍 DEBUG: Top results count: {len(analysis_results.get('top_results', []))}")
            
            # Analyze path widths
            top_results = analysis_results['top_results']
            if top_results:
                print(f"🔍 DEBUG: Found {len(top_results)} paths, analyzing widths...")
                top_results_with_width = self.width_analyzer.analyze_multiple_paths(
                    self.current_grid, top_results
                )
            else:
                print("🔍 DEBUG: No paths found in top_results")
                # Check if there are any results at all
                all_results = analysis_results.get('all_results', [])
                print(f"🔍 DEBUG: All results count: {len(all_results)}")
                if all_results:
                    print(f"🔍 DEBUG: First result keys: {all_results[0].keys() if all_results else 'None'}")
                top_results_with_width = []
            
            return {
                'analysis_results': analysis_results,
                'top_results_with_width': top_results_with_width,
                'start': start,
                'goals': goals,
                'scale_pix_to_m': request.scale_pix_to_m
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error during path planning: {str(e)}")
    
    def create_visualizations(self, planning_data: Dict[str, Any]) -> tuple[Dict[str, str], str]:
        """Create visualization images, save to results folder, and return as base64 strings.
        
        Returns:
            tuple: (visualizations dict, results_directory path)
        """
        visualizations = {}
        results_dir = ""
        try:
            top_results = planning_data['top_results_with_width']
            start = planning_data['start']
            goals = planning_data['goals']
            scale_pix_to_m = planning_data['scale_pix_to_m']
            analysis_results = planning_data['analysis_results']
            
            # Create results directory with timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = os.path.join("results", f"path_planning_{timestamp}")
            os.makedirs(results_dir, exist_ok=True)
            
            print(f"💾 Saving visualizations to: {results_dir}/")
            
            # 1. Grid-based path visualization
            fig_grid = self.visualizer.plot_grid_paths(
                self.current_grid, top_results, start, goals, 
                scale_pix_to_m=scale_pix_to_m, show_width_info=True
            )
            # Save to file
            grid_path = os.path.join(results_dir, "grid_paths.png")
            fig_grid.savefig(grid_path, dpi=150, bbox_inches='tight')
            # Also convert to base64 for API response
            visualizations['grid_paths'] = self._fig_to_base64(fig_grid)
            plt.close(fig_grid)
            print(f"   ✅ Saved: grid_paths.png")
            
            # 2. Algorithm performance plot
            if 'algorithm_performance' in analysis_results:
                fig_perf = self.visualizer.plot_algorithm_performance(
                    analysis_results['algorithm_performance']
                )
                # Save to file
                perf_path = os.path.join(results_dir, "algorithm_performance.png")
                fig_perf.savefig(perf_path, dpi=150, bbox_inches='tight')
                # Convert to base64
                visualizations['algorithm_performance'] = self._fig_to_base64(fig_perf)
                plt.close(fig_perf)
                print(f"   ✅ Saved: algorithm_performance.png")
            
            # 3. Width analysis plots (if width data available)
            if any(r.get('width_stats', {}).get('min_width', 0) > 0 for r in top_results):
                fig_width = self.visualizer.plot_width_analysis(top_results)
                # Save to file
                width_path = os.path.join(results_dir, "width_analysis.png")
                fig_width.savefig(width_path, dpi=150, bbox_inches='tight')
                # Convert to base64
                visualizations['width_analysis'] = self._fig_to_base64(fig_width)
                plt.close(fig_width)
                print(f"   ✅ Saved: width_analysis.png")
            
            # 4. Real-world overlay (if real image available)
            if self.current_real_image is not None:
                try:
                    coordinate_mapper = self.visualizer.create_coordinate_mapper(
                        self.current_grid.shape,
                        self.current_real_image.shape[:2],
                        flip_y=False
                    )
                    
                    # Save real image temporarily
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                        Image.fromarray(self.current_real_image).save(tmp_file.name)
                        
                        fig_real = self.visualizer.plot_real_world_overlay(
                            tmp_file.name, top_results, start, goals, scale_pix_to_m,
                            coordinate_mapper, self.current_grid.shape
                        )
                        # Save to file - THIS IS THE MAIN OUTPUT!
                        real_path = os.path.join(results_dir, "real_world_overlay.png")
                        fig_real.savefig(real_path, dpi=150, bbox_inches='tight')
                        # Convert to base64
                        visualizations['real_world_overlay'] = self._fig_to_base64(fig_real)
                        plt.close(fig_real)
                        print(f"   🎯 Saved: real_world_overlay.png (PATHS ON SATELLITE IMAGE!)")
                        
                        # Clean up temp file
                        os.unlink(tmp_file.name)
                        
                except Exception as e:
                    print(f"Warning: Could not create real-world overlay: {e}")
            
            # Save metadata about the results
            metadata = {
                "timestamp": timestamp,
                "start_point": start,
                "goal_points": goals,
                "scale_pix_to_m": scale_pix_to_m,
                "num_paths_found": len(top_results),
                "results_directory": results_dir
            }
            metadata_path = os.path.join(results_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"   📋 Saved: metadata.json")
            print(f"\n✨ All visualizations saved to: {results_dir}/")
            
            return visualizations, results_dir
            
        except Exception as e:
            print(f"Error creating visualizations: {e}")
            return {}, ""
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string."""
        try:
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            buffer.close()
            return image_base64
        except Exception as e:
            print(f"Error converting figure to base64: {e}")
            return ""

# Initialize service
path_service = PathPlanningService()

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Path Planning API",
        "version": "1.0.0",
        "description": "Multi-goal path planning with real-world mapping and visualization",
        "endpoints": {
            "POST /plan_path": "Main path planning endpoint with real-world overlay",
            "GET /health": "Health check endpoint",
            "GET /algorithms": "List available path planning algorithms",
            "POST /validate_coordinates": "Validate coordinate inputs",
            "GET /docs": "Interactive API documentation",
            "GET /redoc": "Alternative API documentation"
        },
        "usage": {
            "segmentation_image": "Upload binary image (white=roads, black=obstacles)",
            "real_world_image": "Upload satellite/aerial image for path overlay",
            "coordinates": "Provide pixel coordinates for start and goal points"
        }
    }

@app.post("/plan_path", response_model=PathPlanningResponse)
async def plan_path(
    request_data: str = Form(...),
    segmentation_image: UploadFile = File(...),
    real_world_image: Optional[UploadFile] = File(None)
):
    """
    Main path planning endpoint.
    
    Args:
        request_data: JSON string containing PathPlanningRequest
        segmentation_image: Binary segmentation image (roads=white, obstacles=black)
        real_world_image: Optional real-world satellite/aerial image for overlay
    
    Returns:
        PathPlanningResponse with results and visualizations
    """
    try:
        # Validate file types
        if not segmentation_image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Segmentation file must be an image")
        
        if real_world_image and not real_world_image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Real world file must be an image")
        
        # Parse request data
        try:
            print(f"🔍 DEBUG: Raw request_data: {request_data}")
            request = PathPlanningRequest.model_validate_json(request_data)
            print(f"🔍 DEBUG: Parsed request - Start: {request.start_point}, Goals: {len(request.goal_points)} points")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid request data: {str(e)}")
        
        # Validate coordinate bounds will be checked after loading grid
        
        # Load segmentation image
        seg_image_data = await segmentation_image.read()
        if len(seg_image_data) == 0:
            raise HTTPException(status_code=400, detail="Segmentation image is empty")
            
        grid = path_service.load_segmentation_image(seg_image_data)
        path_service.current_grid = grid
        
        # Validate coordinates are within grid bounds
        height, width = grid.shape
        
        # Check start point
        if not (0 <= request.start_point.x < width and 0 <= request.start_point.y < height):
            raise HTTPException(
                status_code=400, 
                detail=f"Start point ({request.start_point.x}, {request.start_point.y}) is outside grid bounds (0-{width-1}, 0-{height-1})"
            )
        
        # Check goal points
        for i, goal in enumerate(request.goal_points):
            if not (0 <= goal.x < width and 0 <= goal.y < height):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Goal point {i+1} ({goal.x}, {goal.y}) is outside grid bounds (0-{width-1}, 0-{height-1})"
                )
        
        # Load real-world image if provided
        if real_world_image:
            real_image_data = await real_world_image.read()
            path_service.current_real_image = path_service.load_real_world_image(real_image_data)
        else:
            path_service.current_real_image = None
        
        # Initialize planners
        path_service.initialize_planners(grid)
        
        # Run path planning
        planning_data = path_service.run_path_planning(request)
        
        # Create visualizations and get results directory
        viz_result = path_service.create_visualizations(planning_data)
        print(f"🔍 DEBUG: viz_result type = {type(viz_result)}, value = {viz_result if isinstance(viz_result, str) else 'tuple' if isinstance(viz_result, tuple) else 'other'}")
        
        if isinstance(viz_result, tuple):
            visualizations, results_directory = viz_result
            print(f"🔍 DEBUG: Unpacked - visualizations type = {type(visualizations)}, results_directory = {results_directory}")
        else:
            raise ValueError(f"Expected tuple, got {type(viz_result)}")
        
        # Format results
        results = []
        top_results = planning_data['top_results_with_width']
        scale_pix_to_m = planning_data['scale_pix_to_m']
        
        for i, result in enumerate(top_results):
            path_result = PathResult(
                path_id=i + 1,
                algorithm=result.get('algorithm', 'Unknown'),
                distance_pixels=result.get('dist', 0.0),
                distance_meters=result.get('dist', 0.0) * scale_pix_to_m,
                path_coordinates=list(zip(result.get('rx', []), result.get('ry', []))),
                width_stats=result.get('width_stats', {})
            )
            results.append(path_result)
        
        # Create response
        response = PathPlanningResponse(
            success=True,
            message=f"Successfully found {len(results)} optimal paths. Results saved to {results_directory}",
            total_paths_found=len(results),
            total_combinations_tested=planning_data['analysis_results'].get('total_combinations_tested', 0),
            results=results,
            algorithm_performance=planning_data['analysis_results'].get('algorithm_performance', {}),
            visualization_images=visualizations,
            results_directory=results_directory
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        # Log full traceback for debugging
        print(f"Unexpected error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Path Planning API"}

@app.get("/algorithms")
async def get_available_algorithms():
    """Get list of available path planning algorithms."""
    try:
        # Create a dummy grid to get available algorithms
        dummy_grid = np.ones((10, 10))
        planners = PlannerFactory.create_planners_dict(dummy_grid)
        return {"algorithms": list(planners.keys())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting algorithms: {str(e)}")

@app.post("/validate_coordinates")
async def validate_coordinates(
    coordinates: List[CoordinatePoint],
    segmentation_image: UploadFile = File(...)
):
    """
    Validate if given coordinates are within valid (passable) areas of the segmentation image.
    
    Args:
        coordinates: List of coordinates to validate
        segmentation_image: Binary segmentation image
    
    Returns:
        Validation results for each coordinate
    """
    try:
        # Load segmentation image
        seg_image_data = await segmentation_image.read()
        grid = path_service.load_segmentation_image(seg_image_data)
        
        results = []
        for i, coord in enumerate(coordinates):
            x, y = int(coord.x), int(coord.y)
            
            # Check bounds
            if 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]:
                is_valid = bool(grid[y, x] == 1)  # 1 = passable road
                message = "Valid coordinate" if is_valid else "Coordinate is in obstacle area"
            else:
                is_valid = False
                message = "Coordinate is out of bounds"
            
            results.append({
                "coordinate_index": i,
                "x": coord.x,
                "y": coord.y,
                "is_valid": is_valid,
                "message": message
            })
        
        return {"validation_results": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error validating coordinates: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting FastAPI Path Planning Server...")
    print("📍 Server: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("🔧 ReDoc: http://localhost:8000/redoc")
    print("⚡ Press Ctrl+C to stop")
    uvicorn.run("fastapi_path_planning:app", host="0.0.0.0", port=8000, reload=True)