#!/bin/bash

echo "🗺️  FIXED cURL Commands for FastAPI Path Planning"
echo "================================================="
echo ""

# Get current directory
CURRENT_DIR=$(pwd)
RESOURCE_DIR="/Users/worakanlasudee/Documents/GitHub/fastapi_seg_plan/FloodNav-Path-Planning/resource"

echo "📁 Current directory: $CURRENT_DIR"
echo "📁 Resource directory: $RESOURCE_DIR"
echo ""

# Check if files exist
if [ -f "$RESOURCE_DIR/map.png" ]; then
    echo "✅ Found: $RESOURCE_DIR/map.png"
else
    echo "❌ Missing: $RESOURCE_DIR/map.png"
fi

if [ -f "$RESOURCE_DIR/satellite.png" ]; then
    echo "✅ Found: $RESOURCE_DIR/satellite.png"
else
    echo "❌ Missing: $RESOURCE_DIR/satellite.png"
fi

echo ""
echo "🚀 OPTION 1: Using Full Paths (Recommended)"
echo "-------------------------------------------"

cat << 'EOF'
curl -X POST "http://localhost:8000/plan_path" \
  -F 'request_data={"start_point":{"x":454,"y":368},"goal_points":[{"x":852,"y":460},{"x":234,"y":830},{"x":302,"y":990},{"x":234,"y":128}],"scale_pix_to_m":0.05,"k_top_paths":3,"hausdorff_tolerance":10.0}' \
  -F "segmentation_image=@/Users/worakanlasudee/Documents/GitHub/fastapi_seg_plan/FloodNav-Path-Planning/resource/map.png" \
  -F "real_world_image=@/Users/worakanlasudee/Documents/GitHub/fastapi_seg_plan/FloodNav-Path-Planning/resource/satellite.png"
EOF

echo ""
echo "🚀 OPTION 2: Copy Files to Current Directory"
echo "--------------------------------------------"
echo "# First copy the files:"
echo "cp $RESOURCE_DIR/map.png ."
echo "cp $RESOURCE_DIR/satellite.png ."
echo ""
echo "# Then run:"
cat << 'EOF'
curl -X POST "http://localhost:8000/plan_path" \
  -F 'request_data={"start_point":{"x":454,"y":368},"goal_points":[{"x":852,"y":460},{"x":234,"y":830},{"x":302,"y":990},{"x":234,"y":128}],"scale_pix_to_m":0.05,"k_top_paths":3,"hausdorff_tolerance":10.0}' \
  -F "segmentation_image=@map.png" \
  -F "real_world_image=@satellite.png"
EOF

echo ""
echo "🚀 OPTION 3: Change to Resource Directory"
echo "-----------------------------------------"
echo "# Change directory first:"
echo "cd $RESOURCE_DIR"
echo ""
echo "# Then run:"
cat << 'EOF'
curl -X POST "http://localhost:8000/plan_path" \
  -F 'request_data={"start_point":{"x":454,"y":368},"goal_points":[{"x":852,"y":460},{"x":234,"y":830},{"x":302,"y":990},{"x":234,"y":128}],"scale_pix_to_m":0.05,"k_top_paths":3,"hausdorff_tolerance":10.0}' \
  -F "segmentation_image=@map.png" \
  -F "real_world_image=@satellite.png"
EOF

echo ""
echo "💡 SIMPLE TEST (2 goals only with full paths)"
echo "--------------------------------------------"

cat << 'EOF'
curl -X POST "http://localhost:8000/plan_path" \
  -F 'request_data={"start_point":{"x":100,"y":100},"goal_points":[{"x":500,"y":200},{"x":300,"y":600}]}' \
  -F "segmentation_image=@/Users/worakanlasudee/Documents/GitHub/fastapi_seg_plan/FloodNav-Path-Planning/resource/map.png" \
  -F "real_world_image=@/Users/worakanlasudee/Documents/GitHub/fastapi_seg_plan/FloodNav-Path-Planning/resource/satellite.png"
EOF

echo ""
echo "🔧 AUTO-EXECUTE OPTIONS:"
echo "========================"

echo ""
echo "A) Copy files and run simple test:"
read -p "   Execute this? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📋 Copying files..."
    cp "$RESOURCE_DIR/map.png" . 2>/dev/null && echo "✅ Copied map.png" || echo "❌ Failed to copy map.png"
    cp "$RESOURCE_DIR/satellite.png" . 2>/dev/null && echo "✅ Copied satellite.png" || echo "❌ Failed to copy satellite.png"
    
    echo ""
    echo "🚀 Executing cURL command..."
    curl -X POST "http://localhost:8000/plan_path" \
      -F 'request_data={"start_point":{"x":100,"y":100},"goal_points":[{"x":500,"y":200},{"x":300,"y":600}]}' \
      -F "segmentation_image=@map.png" \
      -F "real_world_image=@satellite.png"
fi

echo ""
echo "B) Execute with full paths (main.py coordinates):"
read -p "   Execute this? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 Executing cURL command with full paths..."
    curl -X POST "http://localhost:8000/plan_path" \
      -F 'request_data={"start_point":{"x":454,"y":368},"goal_points":[{"x":852,"y":460},{"x":234,"y":830},{"x":302,"y":990},{"x":234,"y":128}],"scale_pix_to_m":0.05,"k_top_paths":3,"hausdorff_tolerance":10.0}' \
      -F "segmentation_image=@/Users/worakanlasudee/Documents/GitHub/fastapi_seg_plan/FloodNav-Path-Planning/resource/map.png" \
      -F "real_world_image=@/Users/worakanlasudee/Documents/GitHub/fastapi_seg_plan/FloodNav-Path-Planning/resource/satellite.png"
fi

echo ""
echo "🎉 Commands ready! Choose the option that works best for you."
echo "💡 Tip: Option 1 (full paths) is usually the most reliable."