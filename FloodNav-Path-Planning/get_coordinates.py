"""
Quick Coordinate Picker for Path Planning
Simple GUI to click and get pixel coordinates from images.

Usage:
    python get_coordinates.py
    
Then select an image file through the file dialog.
"""

import cv2
import numpy as np
import json
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from datetime import datetime

def pick_coordinates_from_image(image_path=None):
    """
    Interactive coordinate picker with visual feedback.
    
    Args:
        image_path: Path to image file (if None, opens file dialog)
    
    Returns:
        dict: Dictionary containing coordinates and metadata
    """
    
    # If no image path provided, open file dialog
    if image_path is None:
        root = tk.Tk()
        root.withdraw()
        image_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        if not image_path:
            print("❌ No image selected")
            return None
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return None
    
    height, width = image.shape[:2]
    print(f"\n{'='*70}")
    print(f"📸 Image: {Path(image_path).name}")
    print(f"📐 Size: {width}x{height} (width x height)")
    print(f"{'='*70}")
    print("\n🎯 CONTROLS:")
    print("  • Left Click  : Add point")
    print("  • Right Click : Remove last point")
    print("  • 's' key     : Save coordinates")
    print("  • 'c' key     : Clear all points")
    print("  • 'q' or ESC  : Quit")
    print(f"{'='*70}\n")
    
    # State variables
    coordinates = []
    display_image = image.copy()
    window_name = "Coordinate Picker - 's' to save, 'c' to clear, 'q' to quit"
    
    def update_display():
        """Redraw the image with all points"""
        nonlocal display_image
        display_image = image.copy()
        
        # Draw connecting lines
        if len(coordinates) > 1:
            for i in range(len(coordinates) - 1):
                pt1 = tuple(coordinates[i])
                pt2 = tuple(coordinates[i + 1])
                cv2.line(display_image, pt1, pt2, (255, 255, 0), 2)
        
        # Draw points
        for i, (x, y) in enumerate(coordinates):
            if i == 0:
                # Start point - Green
                color = (0, 255, 0)
                label = "START"
            else:
                # Goal points - Red
                color = (0, 0, 255)
                label = f"G{i}"
            
            # Draw marker
            cv2.circle(display_image, (x, y), 10, color, -1)
            cv2.circle(display_image, (x, y), 10, (255, 255, 255), 2)
            
            # Draw label
            cv2.putText(display_image, label, (x + 15, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(display_image, f"({x},{y})", (x + 15, y + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow(window_name, display_image)
    
    def mouse_callback(event, x, y, flags, param):
        """Handle mouse events"""
        if event == cv2.EVENT_LBUTTONDOWN:
            coordinates.append([x, y])
            point_type = "START" if len(coordinates) == 1 else f"GOAL{len(coordinates)-1}"
            print(f"✅ {point_type}: (x={x}, y={y})")
            update_display()
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            if coordinates:
                removed = coordinates.pop()
                print(f"❌ Removed: (x={removed[0]}, y={removed[1]})")
                update_display()
        
        elif event == cv2.EVENT_MOUSEMOVE:
            # Show cursor position
            temp = display_image.copy()
            cv2.line(temp, (x, 0), (x, height), (150, 150, 150), 1)
            cv2.line(temp, (0, y), (width, y), (150, 150, 150), 1)
            cv2.putText(temp, f"({x},{y})", (x + 10, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                       cv2.LINE_AA)
            cv2.imshow(window_name, temp)
    
    def save_coordinates():
        """Save coordinates to JSON file"""
        if not coordinates:
            print("⚠️  No coordinates to save")
            return None
        
        # Create output directory
        output_dir = Path("coordinates_output")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = Path(image_path).stem
        output_file = output_dir / f"coords_{image_name}_{timestamp}.json"
        
        # Format data
        data = {
            "image_path": str(image_path),
            "image_size": {"width": width, "height": height},
            "timestamp": timestamp,
            "coordinates": coordinates,
            "start_point": {"x": coordinates[0][0], "y": coordinates[0][1]} if coordinates else None,
            "goal_points": [{"x": c[0], "y": c[1]} for c in coordinates[1:]] if len(coordinates) > 1 else [],
            
            # FastAPI format
            "fastapi_request": {
                "start_point": {"x": coordinates[0][0], "y": coordinates[0][1]} if coordinates else None,
                "goal_points": [{"x": c[0], "y": c[1]} for c in coordinates[1:]] if len(coordinates) > 1 else [],
                "scale_pix_to_m": 0.05,
                "k_top_paths": 3,
                "hausdorff_tolerance": 10.0
            }
        }
        
        # Save
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"💾 Saved: {output_file}")
        print(f"📍 Points: {len(coordinates)} total")
        if coordinates:
            print(f"   START: ({coordinates[0][0]}, {coordinates[0][1]})")
            for i, (x, y) in enumerate(coordinates[1:], 1):
                print(f"   GOAL{i}: ({x}, {y})")
        print(f"{'='*70}\n")
        
        return data
    
    # Setup window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(1200, width), min(800, height))
    cv2.setMouseCallback(window_name, mouse_callback)
    update_display()
    
    # Main loop
    result_data = None
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # Quit
            break
        elif key == ord('s'):  # Save
            result_data = save_coordinates()
        elif key == ord('c'):  # Clear
            coordinates.clear()
            print("🗑️  Cleared all points")
            update_display()
    
    cv2.destroyAllWindows()
    
    # Return final data
    if coordinates and result_data is None:
        # Auto-save if user quit without saving
        result_data = save_coordinates()
    
    return result_data

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🎯 COORDINATE PICKER FOR PATH PLANNING")
    print("="*70)
    
    result = pick_coordinates_from_image()
    
    if result:
        print("\n✅ Coordinates captured successfully!")
        print("\n📋 FastAPI Request Data:")
        print(json.dumps(result['fastapi_request'], indent=2))
    else:
        print("\n❌ No coordinates captured")
