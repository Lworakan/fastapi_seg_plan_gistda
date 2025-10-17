"""
Minimal Model Test - Step by Step Debugging
This script tests each component separately to isolate the segmentation fault.
"""

import os
import sys
import numpy as np

print("="*70)
print("🔬 MINIMAL MODEL DEBUGGING")
print("="*70)

# Test 1: Basic imports
print("📦 Step 1: Testing basic imports...")
try:
    import cv2
    print("   ✅ OpenCV imported")
except Exception as e:
    print(f"   ❌ OpenCV failed: {e}")
    sys.exit(1)

try:
    import numpy as np
    print("   ✅ NumPy imported")
except Exception as e:
    print(f"   ❌ NumPy failed: {e}")
    sys.exit(1)

# Test 2: TensorFlow import with careful memory handling
print("\n📦 Step 2: Testing TensorFlow import...")
try:
    import tensorflow as tf
    print(f"   ✅ TensorFlow {tf.__version__} imported")
    
    # Set memory growth to avoid GPU memory issues
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"   ✅ GPU memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"   ⚠️  GPU config warning: {e}")
    else:
        print("   ℹ️  No GPUs detected, using CPU")
        
except Exception as e:
    print(f"   ❌ TensorFlow failed: {e}")
    sys.exit(1)

# Test 3: Check model file
print("\n📁 Step 3: Testing model file...")
model_path = "models/v5_road_extraction_beefy_unet.h5"
if os.path.exists(model_path):
    size = os.path.getsize(model_path) / (1024*1024)
    print(f"   ✅ Model file exists: {size:.1f} MB")
else:
    print(f"   ❌ Model file not found: {model_path}")
    sys.exit(1)

# Test 4: Check if it's a valid HDF5 file
print("\n🔍 Step 4: Testing HDF5 file validity...")
try:
    import h5py
    with h5py.File(model_path, 'r') as f:
        print(f"   ✅ Valid HDF5 file with keys: {list(f.keys())}")
except Exception as e:
    print(f"   ❌ HDF5 error: {e}")
    sys.exit(1)

# Test 5: Try to load model in the safest way possible
print("\n🔄 Step 5: Attempting minimal model load...")
try:
    # Force CPU usage
    with tf.device('/CPU:0'):
        # Try loading without compilation first
        print("   🔄 Loading model (compile=False)...")
        model = tf.keras.models.load_model(model_path, compile=False)
        print(f"   ✅ Model loaded successfully!")
        print(f"   📐 Input shape: {model.input_shape}")
        print(f"   📐 Output shape: {model.output_shape}")
        
        # Clear the model from memory immediately
        del model
        print("   🗑️  Model cleared from memory")
        
except Exception as e:
    print(f"   ❌ Model loading failed: {e}")
    print(f"   🔍 Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test image loading
print("\n🖼️  Step 6: Testing image loading...")
image_path = "test_images/imagepan2.png"
try:
    if os.path.exists(image_path):
        image = cv2.imread(image_path)
        if image is not None:
            print(f"   ✅ Image loaded: {image.shape}")
            print(f"   📊 Image stats: min={image.min()}, max={image.max()}, dtype={image.dtype}")
        else:
            print(f"   ❌ Failed to decode image")
    else:
        print(f"   ❌ Image not found: {image_path}")
except Exception as e:
    print(f"   ❌ Image loading error: {e}")

print("\n" + "="*70)
print("✅ ALL BASIC TESTS PASSED!")
print("="*70)
print("\n💡 If you see this message, the segmentation fault is likely")
print("   happening during model prediction, not during loading.")
print("\n🔧 Next steps:")
print("   1. Try a smaller test image")
print("   2. Check if the model architecture is compatible")
print("   3. Try running with different TensorFlow versions")
print("="*70)