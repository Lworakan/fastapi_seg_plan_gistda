"""
Quick Test Script for UNet Model
Tests your specific model with your image.

Usage:
    python quick_test.py
"""

import os
import sys
from pathlib import Path

print("="*70)
print("🧪 QUICK UNET MODEL TEST")
print("="*70)

# Check if model exists
model_path = "models/unet_road_segmentation.h5"
if not os.path.exists(model_path):
    print(f"❌ Model not found: {model_path}")
    print("   Please place your model file in the models/ directory")
    sys.exit(1)
else:
    print(f"✅ Model found: {model_path}")

# Check if test image exists
test_image = "test_images/imagepan3.png"
if not os.path.exists(test_image):
    print(f"❌ Test image not found: {test_image}")
    print("   Please place your test image in the test_images/ directory")
    sys.exit(1)
else:
    print(f"✅ Test image found: {test_image}")

print(f"\n{'='*70}")
print("🚀 Starting evaluation...")
print("="*70 + "\n")

# Import and run evaluation
try:
    from eval_unet import UNetEvaluator
    
    # Create evaluator
    print("📦 Loading model...")
    evaluator = UNetEvaluator(
        model_path=model_path,
        input_size=(128, 128),  # Match model's expected input size
        dataset_name="DeepGlobe"
    )
    
    # Evaluate single image
    print("🔍 Running inference...")
    result = evaluator.evaluate_image(
        image_path=test_image,
        save_dir="results/quick_test"
    )
    
    print(f"\n{'='*70}")
    print("✅ EVALUATION COMPLETE!")
    print("="*70)
    print(f"📐 Input shape: {result['input_shape']}")
    print(f"📐 Output shape: {result['output_shape']}")
    print(f"\n📁 Results saved to:")
    for key, path in result['saved_files'].items():
        print(f"   • {key}: {path}")
    print("="*70 + "\n")
    
    print("💡 TIP: Open the overlay image to see the segmentation results!")
    print(f"   {result['saved_files']['overlay']}")

except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("\n💡 Install required packages:")
    print("   pip install -r requirements.txt")
    sys.exit(1)

except Exception as e:
    print(f"❌ Error during evaluation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
