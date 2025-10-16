#!/bin/bash
# Fixed cURL Commands for FastAPI Path Planning

echo "🔧 FIXED cURL COMMANDS"
echo "====================="

# Navigate to the correct directory first
cd /Users/worakanlasudee/Documents/GitHub/fastapi_seg_plan/FloodNav-Path-Planning/resource

echo "📍 Current directory: $(pwd)"
echo "📁 Available files:"
ls -la *.png 2>/dev/null || echo "No PNG files found"

echo ""
echo "✅ WORKING cURL COMMAND (Copy & Paste This!):"
echo "============================================="

echo 'curl -X POST "http://localhost:8000/plan_path" \'
echo '  -F '\''request_data={"start_point":{"x":454,"y":368},"goal_points":[{"x":852,"y":460},{"x":234,"y":830},{"x":302,"y":990},{"x":234,"y":128}],"scale_pix_to_m":0.05,"k_top_paths":3,"hausdorff_tolerance":10.0}'\'' \'
echo '  -F "segmentation_image=@map.png" \'
echo '  -F "real_world_image=@satellite.png"'

echo ""
echo "🔧 TECHNICAL VERSION (for reference):"
echo "------------------------------------"

# Fixed version with proper file paths and JSON escaping
cat << 'EOF'
# WORKING CURL COMMAND (Fixed JSON formatting)
curl -X POST 'http://localhost:8000/plan_path' \
  -F 'request_data={"start_point":{"x":454,"y":368},"goal_points":[{"x":852,"y":460},{"x":234,"y":830},{"x":302,"y":990},{"x":234,"y":128}],"scale_pix_to_m":0.05,"k_top_paths":3,"hausdorff_tolerance":10.0}' \
  -F 'segmentation_image=@map.png' \
  -F 'real_world_image=@satellite.png'
EOF

echo ""
echo "💡 SIMPLE TEST (2 goals only):"
echo "-----------------------------"

cat << 'EOF'
curl -X POST 'http://localhost:8000/plan_path' \
  -F 'request_data={"start_point":{"x":454,"y":368},"goal_points":[{"x":852,"y":460},{"x":234,"y":830}]}' \
  -F 'segmentation_image=@map.png' \
  -F 'real_world_image=@satellite.png'
EOF

echo ""
echo "🔧 TROUBLESHOOTING COMMAND:"
echo "-------------------------"
echo "# Test with minimal data first:"

cat << 'EOF'
curl -X POST 'http://localhost:8000/plan_path' \
  -F 'request_data={"start_point":{"x":100,"y":100},"goal_points":[{"x":200,"y":200}]}' \
  -F 'segmentation_image=@map.png'
EOF

echo ""
echo "📋 STEP-BY-STEP DEBUGGING:"
echo "========================="
echo "1. Check if files exist:"
echo "   ls -la map.png satellite.png"
echo ""
echo "2. Test API health:"
echo "   curl http://localhost:8000/health"
echo ""
echo "3. Check API docs:"
echo "   open http://localhost:8000/docs"
echo ""
echo "4. Run this script:"
echo "   bash fixed_curl_commands.sh"