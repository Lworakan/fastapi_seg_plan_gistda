#!/usr/bin/env python3
"""
Example Plan Path Data Input for FastAPI Path Planning
Uses existing data from the resource folder
"""

import requests
import json
import base64
import os

def create_example_with_existing_data():
    """Create example request using existing data in resource folder."""
    
    # API endpoint
    api_url = "http://localhost:8000/plan_path"
    
    # Paths to your existing data files
    resource_dir = "/Users/worakanlasudee/Documents/GitHub/fastapi_seg_plan/FloodNav-Path-Planning/resource"
    
    # File paths
    segmentation_image_path = os.path.join(resource_dir, "map.png")
    real_world_image_path = os.path.join(resource_dir, "satellite.png")
    
    # Alternative files you can try
    alternative_files = {
        "segmentation": [
            os.path.join(resource_dir, "map.png"),
            os.path.join(resource_dir, "map2.png")
        ],
        "real_world": [
            os.path.join(resource_dir, "satellite.png"),
            os.path.join(resource_dir, "satellite2.png")
        ]
    }
    
    print("🗺️  FastAPI Path Planning - Example with Existing Data")
    print("=" * 60)
    
    # Check which files exist
    print("📁 Available files:")
    for seg_file in alternative_files["segmentation"]:
        status = "✅" if os.path.exists(seg_file) else "❌"
        print(f"   {status} Segmentation: {os.path.basename(seg_file)}")
    
    for real_file in alternative_files["real_world"]:
        status = "✅" if os.path.exists(real_file) else "❌"
        print(f"   {status} Real World: {os.path.basename(real_file)}")
    
    if not os.path.exists(segmentation_image_path):
        print(f"❌ Segmentation image not found: {segmentation_image_path}")
        return
    
    # Example 1: Using coordinates from main.py
    print("\n" + "="*50)
    print("EXAMPLE 1: Using Main.py Coordinates")
    print("="*50)
    
    request_data_1 = {
        "start_point": {"x": 454, "y": 368},
        "goal_points": [
            {"x": 852, "y": 460},  # point['1']
            {"x": 234, "y": 830},  # point['2'] 
            {"x": 302, "y": 990},  # point['3']
            {"x": 234, "y": 128}   # point['4']
        ],
        "scale_pix_to_m": 0.05,
        "k_top_paths": 3,
        "hausdorff_tolerance": 10.0
    }
    
    print("📍 Start point:", request_data_1["start_point"])
    print("🎯 Goal points:", len(request_data_1["goal_points"]), "destinations")
    
    # Example 2: Simplified coordinates
    print("\n" + "="*50)
    print("EXAMPLE 2: Simplified Coordinates")
    print("="*50)
    
    request_data_2 = {
        "start_point": {"x": 100, "y": 100},
        "goal_points": [
            {"x": 500, "y": 200},
            {"x": 300, "y": 600}
        ],
        "scale_pix_to_m": 0.05,
        "k_top_paths": 2,
        "hausdorff_tolerance": 15.0
    }
    
    print("📍 Start point:", request_data_2["start_point"])
    print("🎯 Goal points:", len(request_data_2["goal_points"]), "destinations")
    
    # Example 3: High precision coordinates
    print("\n" + "="*50)
    print("EXAMPLE 3: High Precision Setup")
    print("="*50)
    
    request_data_3 = {
        "start_point": {"x": 200.5, "y": 150.3},
        "goal_points": [
            {"x": 800.7, "y": 400.2},
            {"x": 450.1, "y": 750.8},
            {"x": 600.4, "y": 300.6}
        ],
        "scale_pix_to_m": 0.02,  # Higher precision: 2cm per pixel
        "k_top_paths": 5,
        "hausdorff_tolerance": 12.0
    }
    
    print("📍 Start point:", request_data_3["start_point"])
    print("🎯 Goal points:", len(request_data_3["goal_points"]), "destinations")
    print("🔍 Scale: 2cm per pixel (high precision)")
    
    # Choose which example to run
    examples = {
        "1": ("Main.py Coordinates", request_data_1),
        "2": ("Simplified", request_data_2), 
        "3": ("High Precision", request_data_3)
    }
    
    print(f"\n📋 Choose example to run:")
    for key, (name, _) in examples.items():
        print(f"   {key}: {name}")
    
    choice = input("\nEnter choice (1-3) or 'all' for all examples: ").strip()
    
    if choice.lower() == 'all':
        chosen_examples = examples.values()
    else:
        chosen_examples = [examples.get(choice, examples["1"])]
    
    # Run chosen examples
    for example_name, request_data in chosen_examples:
        print(f"\n🚀 Running: {example_name}")
        print("-" * 40)
        
        # Prepare files
        files = {
            'request_data': ('', json.dumps(request_data)),
            'segmentation_image': ('seg.png', open(segmentation_image_path, 'rb'), 'image/png')
        }
        
        # Add real world image if it exists
        if os.path.exists(real_world_image_path):
            files['real_world_image'] = ('real.png', open(real_world_image_path, 'rb'), 'image/png')
            print("✅ Using real world overlay")
        else:
            print("⚠️  No real world image - grid visualization only")
        
        try:
            print("📤 Sending request to API...")
            response = requests.post(api_url, files=files)
            
            if response.status_code == 200:
                result = response.json()
                
                if result['success']:
                    print(f"✅ SUCCESS: Found {result['total_paths_found']} paths!")
                    print(f"🔍 Tested {result['total_combinations_tested']} combinations")
                    
                    # Display path results
                    for i, path in enumerate(result['results']):
                        print(f"\n  Path {i+1}: {path['algorithm']}")
                        print(f"    Distance: {path['distance_pixels']:.1f} pixels ({path['distance_meters']:.1f}m)")
                        print(f"    Points: {len(path['path_coordinates'])} coordinates")
                        
                        if path['width_stats']:
                            width_stats = path['width_stats']
                            min_w = width_stats.get('min_width', 'N/A')
                            max_w = width_stats.get('max_width', 'N/A')
                            print(f"    Width: {min_w} - {max_w} pixels")
                    
                    # Save visualizations
                    output_dir = f"example_results_{example_name.lower().replace(' ', '_')}"
                    os.makedirs(output_dir, exist_ok=True)
                    
                    print(f"\n💾 Saving results to: {output_dir}/")
                    for viz_name, base64_img in result['visualization_images'].items():
                        img_bytes = base64.b64decode(base64_img)
                        output_path = os.path.join(output_dir, f"{viz_name}.png")
                        
                        with open(output_path, 'wb') as f:
                            f.write(img_bytes)
                        
                        print(f"    📊 {viz_name}.png")
                        
                        if viz_name == 'real_world_overlay':
                            print(f"       🎯 ← Paths drawn on your satellite image!")
                
                else:
                    print(f"❌ Path planning failed: {result['message']}")
                    
            else:
                print(f"❌ API Error {response.status_code}")
                try:
                    error_detail = response.json()
                    print(f"   Details: {error_detail.get('detail', 'Unknown error')}")
                except:
                    print(f"   Response: {response.text[:200]}...")
                    
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to API server!")
            print("💡 Make sure the server is running:")
            print("   cd /Users/worakanlasudee/Documents/GitHub/fastapi_seg_plan/FloodNav-Path-Planning")
            print("   python3 fastapi_path_planning.py")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        finally:
            # Close file handles
            for file_info in files.values():
                if hasattr(file_info[1], 'close'):
                    file_info[1].close()
        
        print()

def print_curl_examples():
    """Print cURL command examples."""
    print("\n" + "="*60)
    print("📋 cURL COMMAND EXAMPLES")
    print("="*60)
    
    resource_dir = "/Users/worakanlasudee/Documents/GitHub/fastapi_seg_plan/FloodNav-Path-Planning/resource"
    
    # Example 1: Main.py coordinates
    print("\n🔹 Example 1: Main.py Coordinates")
    print("-" * 35)
    curl_cmd_1 = f'''curl -X POST "http://localhost:8000/plan_path" \\
  -F 'request_data="{{
    \\"start_point\\": {{\\"x\\": 454, \\"y\\": 368}},
    \\"goal_points\\": [
      {{\\"x\\": 852, \\"y\\": 460}},
      {{\\"x\\": 234, \\"y\\": 830}},
      {{\\"x\\": 302, \\"y\\": 990}}
    ],
    \\"scale_pix_to_m\\": 0.05,
    \\"k_top_paths\\": 3,
    \\"hausdorff_tolerance\\": 10.0
  }}"' \\
  -F "segmentation_image=@{resource_dir}/map.png" \\
  -F "real_world_image=@{resource_dir}/satellite.png"'''
    
    print(curl_cmd_1)
    
    # Example 2: Simple coordinates
    print("\n🔹 Example 2: Simple Coordinates")
    print("-" * 32)
    curl_cmd_2 = f'''curl -X POST "http://localhost:8000/plan_path" \\
  -F 'request_data="{{
    \\"start_point\\": {{\\"x\\": 100, \\"y\\": 100}},
    \\"goal_points\\": [
      {{\\"x\\": 500, \\"y\\": 200}},
      {{\\"x\\": 300, \\"y\\": 600}}
    ]
  }}"' \\
  -F "segmentation_image=@{resource_dir}/map.png"'''
    
    print(curl_cmd_2)

def print_json_examples():
    """Print JSON request examples."""
    print("\n" + "="*60)
    print("📋 JSON REQUEST DATA EXAMPLES")
    print("="*60)
    
    examples = {
        "Main.py Coordinates": {
            "start_point": {"x": 454, "y": 368},
            "goal_points": [
                {"x": 852, "y": 460},
                {"x": 234, "y": 830}, 
                {"x": 302, "y": 990},
                {"x": 234, "y": 128}
            ],
            "scale_pix_to_m": 0.05,
            "k_top_paths": 3,
            "hausdorff_tolerance": 10.0
        },
        "Simple Path": {
            "start_point": {"x": 100, "y": 100},
            "goal_points": [
                {"x": 500, "y": 200},
                {"x": 300, "y": 600}
            ]
        },
        "High Precision": {
            "start_point": {"x": 200.5, "y": 150.3},
            "goal_points": [
                {"x": 800.7, "y": 400.2},
                {"x": 450.1, "y": 750.8},
                {"x": 600.4, "y": 300.6}
            ],
            "scale_pix_to_m": 0.02,
            "k_top_paths": 5,
            "hausdorff_tolerance": 12.0
        }
    }
    
    for name, data in examples.items():
        print(f"\n🔹 {name}:")
        print("-" * (len(name) + 3))
        print(json.dumps(data, indent=2))

if __name__ == "__main__":
    print("🗺️  FastAPI Path Planning - Example Data Generator")
    print("Using your existing resource files")
    print()
    
    # Run interactive example
    create_example_with_existing_data()
    
    # Print additional examples
    print_json_examples()
    print_curl_examples()
    
    print(f"\n🎉 Complete! Your existing data files are ready to use:")
    print(f"   📁 Segmentation: resource/map.png, resource/map2.png")
    print(f"   🛰️  Real World: resource/satellite.png, resource/satellite2.png")
    print(f"\n💡 API Documentation: http://localhost:8000/docs")