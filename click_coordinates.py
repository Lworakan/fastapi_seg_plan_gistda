"""
Interactive Image Coordinate Picker
Click on an image to get pixel coordinates (x, y) for path planning.

Usage:
    python click_coordinates.py <image_path>
    
    Left Click: Add coordinate point
    Right Click: Remove last point
    Press 's': Save coordinates to JSON file
    Press 'c': Clear all points
    Press 'q' or ESC: Quit
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime

class CoordinatePicker:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        
        self.display_image = self.image.copy()
        self.coordinates = []
        self.window_name = "Click to Pick Coordinates - Press 's' to save, 'c' to clear, 'q' to quit"
        
        # Get image dimensions
        self.height, self.width = self.image.shape[:2]
        print(f"📐 Image size: {self.width}x{self.height} (width x height)")
        print(f"📍 Coordinate range: x=[0, {self.width-1}], y=[0, {self.height-1}]")
        print("\n" + "="*70)
        print("INSTRUCTIONS:")
        print("  • Left Click: Add coordinate point")
        print("  • Right Click: Remove last point")
        print("  • Press 's': Save coordinates to JSON file")
        print("  • Press 'c': Clear all points")
        print("  • Press 'q' or ESC: Quit")
        print("="*70 + "\n")
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Left click - add point
            self.add_point(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click - remove last point
            self.remove_last_point()
        elif event == cv2.EVENT_MOUSEMOVE:
            # Show current mouse position
            self.show_cursor_position(x, y)
    
    def add_point(self, x, y):
        """Add a coordinate point"""
        self.coordinates.append({"x": x, "y": y})
        print(f"✅ Point {len(self.coordinates)}: (x={x}, y={y})")
        self.update_display()
    
    def remove_last_point(self):
        """Remove the last added point"""
        if self.coordinates:
            removed = self.coordinates.pop()
            print(f"❌ Removed point: (x={removed['x']}, y={removed['y']})")
            self.update_display()
        else:
            print("⚠️  No points to remove")
    
    def clear_points(self):
        """Clear all points"""
        self.coordinates = []
        print("🗑️  All points cleared")
        self.update_display()
    
    def show_cursor_position(self, x, y):
        """Show cursor position on image"""
        temp_image = self.display_image.copy()
        
        # Draw crosshair at cursor
        cv2.line(temp_image, (x, 0), (x, self.height), (200, 200, 200), 1)
        cv2.line(temp_image, (0, y), (self.width, y), (200, 200, 200), 1)
        
        # Display coordinates
        text = f"({x}, {y})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Position text near cursor
        text_x = x + 10
        text_y = y - 10
        
        # Keep text within image bounds
        if text_x + text_width > self.width:
            text_x = x - text_width - 10
        if text_y < text_height:
            text_y = y + text_height + 10
        
        # Draw background rectangle
        cv2.rectangle(temp_image, 
                     (text_x - 2, text_y - text_height - 2),
                     (text_x + text_width + 2, text_y + baseline + 2),
                     (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(temp_image, text, (text_x, text_y), 
                   font, font_scale, (255, 255, 255), thickness)
        
        cv2.imshow(self.window_name, temp_image)
    
    def update_display(self):
        """Update the display with current points"""
        self.display_image = self.image.copy()
        
        # Draw all points
        for i, coord in enumerate(self.coordinates):
            x, y = coord['x'], coord['y']
            
            # Different colors for first point (start) and others (goals)
            if i == 0:
                color = (0, 255, 0)  # Green for start point
                label = "START"
            else:
                color = (0, 0, 255)  # Red for goal points
                label = f"GOAL{i}"
            
            # Draw filled circle
            cv2.circle(self.display_image, (x, y), 8, color, -1)
            # Draw outline
            cv2.circle(self.display_image, (x, y), 8, (255, 255, 255), 2)
            
            # Draw label
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.display_image, label, (x + 12, y - 12),
                       font, 0.6, color, 2)
            
            # Draw coordinate text
            coord_text = f"({x},{y})"
            cv2.putText(self.display_image, coord_text, (x + 12, y + 5),
                       font, 0.4, (255, 255, 255), 1)
        
        # Draw lines connecting points
        if len(self.coordinates) > 1:
            for i in range(len(self.coordinates) - 1):
                pt1 = (self.coordinates[i]['x'], self.coordinates[i]['y'])
                pt2 = (self.coordinates[i+1]['x'], self.coordinates[i+1]['y'])
                cv2.line(self.display_image, pt1, pt2, (255, 255, 0), 2)
        
        cv2.imshow(self.window_name, self.display_image)
    
    def save_coordinates(self):
        """Save coordinates to JSON file"""
        if not self.coordinates:
            print("⚠️  No coordinates to save")
            return
        
        # Create output directory
        output_dir = Path("coordinates_output")
        output_dir.mkdir(exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = Path(self.image_path).stem
        output_file = output_dir / f"coordinates_{image_name}_{timestamp}.json"
        
        # Prepare data
        data = {
            "image_path": str(self.image_path),
            "image_size": {
                "width": self.width,
                "height": self.height
            },
            "timestamp": timestamp,
            "start_point": self.coordinates[0] if self.coordinates else None,
            "goal_points": self.coordinates[1:] if len(self.coordinates) > 1 else [],
            "all_coordinates": self.coordinates,
            "fastapi_format": {
                "start_point": self.coordinates[0] if self.coordinates else None,
                "goal_points": self.coordinates[1:] if len(self.coordinates) > 1 else [],
                "scale_pix_to_m": 0.05,
                "k_top_paths": 3,
                "hausdorff_tolerance": 10.0
            }
        }
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n💾 Coordinates saved to: {output_file}")
        print(f"📊 Total points: {len(self.coordinates)}")
        if self.coordinates:
            print(f"   - Start point: (x={self.coordinates[0]['x']}, y={self.coordinates[0]['y']})")
            if len(self.coordinates) > 1:
                print(f"   - Goal points: {len(self.coordinates) - 1}")
                for i, coord in enumerate(self.coordinates[1:], 1):
                    print(f"     Goal {i}: (x={coord['x']}, y={coord['y']})")
        
        # Also print curl command format
        print("\n" + "="*70)
        print("FASTAPI REQUEST FORMAT:")
        print("="*70)
        print(json.dumps(data['fastapi_format'], indent=2))
        print("="*70 + "\n")
        
        return output_file
    
    def run(self):
        """Main loop"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Initial display
        self.update_display()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("\n👋 Exiting...")
                break
            elif key == ord('s'):  # Save
                self.save_coordinates()
            elif key == ord('c'):  # Clear
                self.clear_points()
        
        cv2.destroyAllWindows()
        
        # Print summary
        if self.coordinates:
            print("\n" + "="*70)
            print("FINAL COORDINATES SUMMARY:")
            print("="*70)
            for i, coord in enumerate(self.coordinates):
                point_type = "START" if i == 0 else f"GOAL{i}"
                print(f"  {point_type}: (x={coord['x']}, y={coord['y']})")
            print("="*70 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Interactive image coordinate picker for path planning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Pick coordinates from a segmentation image
  python click_coordinates.py path/to/segmentation.png
  
  # Pick coordinates from satellite image
  python click_coordinates.py path/to/satellite.png

Instructions:
  - First click will be the START point (green)
  - Subsequent clicks will be GOAL points (red)
  - Right-click to remove the last point
  - Press 's' to save coordinates to JSON
  - Press 'c' to clear all points
  - Press 'q' or ESC to quit
        """
    )
    parser.add_argument('image_path', type=str, help='Path to the image file')
    
    args = parser.parse_args()
    
    try:
        picker = CoordinatePicker(args.image_path)
        picker.run()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise

if __name__ == "__main__":
    main()
