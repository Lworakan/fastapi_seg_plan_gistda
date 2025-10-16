#!/bin/bash
# Quick test with all 4 goals from main.py

echo "🧪 Testing FastAPI Path Planning with 4 goals..."
echo ""

curl -X POST "http://localhost:8000/plan_path" \
  -F 'request_data={"start_point":{"x":454,"y":368},"goal_points":[{"x":852,"y":460},{"x":234,"y":830},{"x":302,"y":990},{"x":234,"y":128}],"scale_pix_to_m":0.05,"k_top_paths":3,"hausdorff_tolerance":10.0}' \
  -F "segmentation_image=@resource/map.png" \
  -F "real_world_image=@resource/satellite.png" \
  | python3 -m json.tool

echo ""
echo "✅ Test complete!"
