#!/usr/bin/env python3
"""
Debug FastAPI Path Planning Validation Issues
Test different request formats to identify the problem
"""

import requests
import json
import os

def test_api_endpoints():
    """Test API endpoints to debug validation issues."""
    
    base_url = "http://localhost:8000"
    
    print("🔍 DEBUGGING FASTAPI PATH PLANNING")
    print("=" * 40)
    
    # Test 1: Health check
    print("\n1️⃣ Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print(f"   ✅ Health: {response.json()}")
        else:
            print(f"   ❌ Error: {response.text}")
    except Exception as e:
        print(f"   ❌ Connection error: {e}")
        print("   💡 Make sure FastAPI server is running!")
        return
    
    # Test 2: Root endpoint
    print("\n2️⃣ Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ API: {result.get('message', 'Unknown')}")
            print(f"   📋 Available endpoints: {len(result.get('endpoints', {}))}")
        else:
            print(f"   ❌ Error: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Simple coordinate validation
    print("\n3️⃣ Testing coordinate validation...")
    
    simple_request = {
        "start_point": {"x": 100, "y": 100},
        "goal_points": [{"x": 200, "y": 200}]
    }
    
    # Check if resource files exist
    resource_dir = "/Users/worakanlasudee/Documents/GitHub/fastapi_seg_plan/FloodNav-Path-Planning/resource"
    map_file = os.path.join(resource_dir, "map.png")
    satellite_file = os.path.join(resource_dir, "satellite.png")
    
    print(f"   📁 Checking files:")
    print(f"      Map: {'✅' if os.path.exists(map_file) else '❌'} {map_file}")
    print(f"      Satellite: {'✅' if os.path.exists(satellite_file) else '❌'} {satellite_file}")
    
    if not os.path.exists(map_file):
        print("   ⚠️  Cannot test without map.png file")
        return
    
    # Test 4: Minimal request (segmentation only)
    print("\n4️⃣ Testing minimal request (segmentation only)...")
    
    try:
        files = {
            'request_data': ('', json.dumps(simple_request)),
            'segmentation_image': ('map.png', open(map_file, 'rb'), 'image/png')
        }
        
        response = requests.post(f"{base_url}/plan_path", files=files)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Success: {result['success']}")
            print(f"   🛣️  Paths found: {result.get('total_paths_found', 0)}")
        elif response.status_code == 422:
            try:
                error_detail = response.json()
                print(f"   ❌ Validation Error 422:")
                print(f"      {json.dumps(error_detail, indent=6)}")
            except:
                print(f"   ❌ Validation Error 422: {response.text}")
        else:
            print(f"   ❌ Error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Request error: {e}")
    finally:
        try:
            files['segmentation_image'][1].close()
        except:
            pass
    
    # Test 5: Full request (with real world image)
    if os.path.exists(satellite_file):
        print("\n5️⃣ Testing full request (with real world image)...")
        
        full_request = {
            "start_point": {"x": 454, "y": 368},
            "goal_points": [
                {"x": 852, "y": 460},
                {"x": 234, "y": 830}
            ],
            "scale_pix_to_m": 0.05,
            "k_top_paths": 2,
            "hausdorff_tolerance": 10.0
        }
        
        try:
            files = {
                'request_data': ('', json.dumps(full_request)),
                'segmentation_image': ('map.png', open(map_file, 'rb'), 'image/png'),
                'real_world_image': ('satellite.png', open(satellite_file, 'rb'), 'image/png')
            }
            
            response = requests.post(f"{base_url}/plan_path", files=files)
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Success: {result['success']}")
                print(f"   🛣️  Paths: {result.get('total_paths_found', 0)}")
                print(f"   🖼️  Visualizations: {len(result.get('visualization_images', {}))}")
            elif response.status_code == 422:
                try:
                    error_detail = response.json()
                    print(f"   ❌ Validation Error 422:")
                    print(f"      {json.dumps(error_detail, indent=6)}")
                except:
                    print(f"   ❌ Validation Error 422: {response.text}")
            else:
                print(f"   ❌ Error {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Request error: {e}")
        finally:
            try:
                files['segmentation_image'][1].close()
                files['real_world_image'][1].close()
            except:
                pass
    
    # Test 6: Show correct cURL format
    print("\n6️⃣ Correct cURL command format:")
    print("-" * 35)
    
    # Escape the JSON for cURL
    escaped_json = json.dumps(simple_request).replace('"', '\\"')
    
    curl_command = f'''curl -X POST "{base_url}/plan_path" \\
  -F 'request_data="{escaped_json}"' \\
  -F "segmentation_image=@{map_file}"'''
    
    if os.path.exists(satellite_file):
        curl_command += f' \\\n  -F "real_world_image=@{satellite_file}"'
    
    print(curl_command)
    
    print(f"\n💡 Debugging Tips:")
    print(f"   • Make sure you're in the correct directory")
    print(f"   • Use relative paths: @map.png instead of full paths") 
    print(f"   • Escape JSON quotes properly in cURL")
    print(f"   • Check file permissions and existence")
    print(f"   • Try the interactive docs: {base_url}/docs")

if __name__ == "__main__":
    test_api_endpoints()