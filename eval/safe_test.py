"""
Safe UNet Model Test Script
Includes memory management and error handling for segmentation faults.

Usage:
    python safe_test.py
"""

import os
import sys
import gc
import warnings
from pathlib import Path

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

# Force CPU-only mode to avoid GPU memory issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

print("="*70)
print("🛡️  SAFE UNET MODEL TEST")
print("="*70)
print("🔧 Running in CPU-only mode to avoid GPU issues")

# Check if model exists
model_path = "models/v5_road_extraction_beefy_unet.h5"
if not os.path.exists(model_path):
    print(f"❌ Model not found: {model_path}")
    print("   Available models:")
    models_dir = Path("models")
    if models_dir.exists():
        for model_file in models_dir.glob("*.h5"):
            print(f"   - {model_file.name}")
    sys.exit(1)
else:
    print(f"✅ Model found: {model_path}")

# Check if test image exists
test_image = "test_images/imagepan2.png"
if not os.path.exists(test_image):
    print(f"❌ Test image not found: {test_image}")
    print("   Available images:")
    images_dir = Path("test_images")
    if images_dir.exists():
        for img_file in images_dir.glob("*.png"):
            print(f"   - {img_file.name}")
    sys.exit(1)
else:
    print(f"✅ Test image found: {test_image}")

print(f"\n{'='*70}")
print("🚀 Starting safe evaluation...")
print("="*70 + "\n")

try:
    # Import TensorFlow with memory growth
    import tensorflow as tf
    print(f"✅ TensorFlow version: {tf.__version__}")
    
    # Configure TensorFlow for safety
    tf.config.set_soft_device_placement(True)
    
    # Limit memory growth if GPU is available (even though we're using CPU)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU config warning: {e}")
    
    # Import after TF configuration
    from tensorflow.keras.models import load_model
    import numpy as np
    import cv2
    
    print("📦 Loading model safely...")
    
    # Load model with error handling
    try:
        model = load_model(model_path, compile=False)
        print(f"✅ Model loaded successfully")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        sys.exit(1)
    
    # Load and preprocess image
    print("🖼️  Loading test image...")
    
    image = cv2.imread(test_image)
    if image is None:
        print(f"❌ Could not load image: {test_image}")
        sys.exit(1)
    
    original_shape = image.shape[:2]
    print(f"   Original image shape: {original_shape}")
    
    # Get expected input size from model
    expected_height = model.input_shape[1]
    expected_width = model.input_shape[2]
    print(f"   Model expects: ({expected_height}, {expected_width})")
    
    # Preprocess
    print("🔄 Preprocessing image...")
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to expected size
    image_resized = cv2.resize(image_rgb, (expected_width, expected_height))
    
    # Normalize (using DeepGlobe parameters)
    mean = np.array([70.95, 71.16, 71.31])
    std = np.array([34.00, 35.18, 36.40])
    image_normalized = (image_resized.astype(np.float32) - mean) / std
    
    # Add batch dimension
    image_batch = np.expand_dims(image_normalized, axis=0)
    print(f"   Input tensor shape: {image_batch.shape}")
    
    # Run prediction with memory management
    print("🔍 Running inference...")
    
    try:
        prediction = model.predict(image_batch, verbose=0, batch_size=1)
        print(f"✅ Prediction successful!")
        print(f"   Output shape: {prediction.shape}")
        
        # Process prediction
        if len(prediction.shape) == 4:
            # Remove batch dimension
            pred_mask = prediction[0]
            
            # Handle different output formats
            if pred_mask.shape[-1] == 1:
                # Single channel output
                probability_map = pred_mask[:, :, 0]
                segmentation_mask = (probability_map > 0.5).astype(np.uint8)
            else:
                # Multi-channel output
                segmentation_mask = np.argmax(pred_mask, axis=-1).astype(np.uint8)
                probability_map = pred_mask[:, :, 1] if pred_mask.shape[-1] > 1 else pred_mask[:, :, 0]
        else:
            probability_map = prediction[0]
            segmentation_mask = (probability_map > 0.5).astype(np.uint8)
        
        print(f"   Segmentation mask shape: {segmentation_mask.shape}")
        print(f"   Unique values in mask: {np.unique(segmentation_mask)}")
        
        # Save results
        print("💾 Saving results...")
        
        results_dir = Path("results/safe_test")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save binary mask
        mask_path = results_dir / "segmentation_mask.png"
        mask_image = (segmentation_mask * 255).astype(np.uint8)
        cv2.imwrite(str(mask_path), mask_image)
        
        # Save probability map
        prob_path = results_dir / "probability_map.png"
        prob_image = (probability_map * 255).astype(np.uint8)
        prob_colored = cv2.applyColorMap(prob_image, cv2.COLORMAP_JET)
        cv2.imwrite(str(prob_path), prob_colored)
        
        # Create overlay
        overlay_path = results_dir / "overlay.png"
        
        # Resize mask to original image size
        if original_shape != segmentation_mask.shape:
            mask_resized = cv2.resize(segmentation_mask, (original_shape[1], original_shape[0]), 
                                     interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = segmentation_mask
        
        # Create colored overlay
        colored_mask = np.zeros((mask_resized.shape[0], mask_resized.shape[1], 3), dtype=np.uint8)
        colored_mask[mask_resized == 1] = [0, 255, 0]  # Green for roads
        
        # Blend with original
        overlay = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
        cv2.imwrite(str(overlay_path), overlay)
        
        print(f"\n{'='*70}")
        print("✅ EVALUATION COMPLETE!")
        print("="*70)
        print(f"📁 Results saved to: {results_dir}")
        print(f"   • Binary mask: {mask_path}")
        print(f"   • Probability map: {prob_path}")
        print(f"   • Overlay: {overlay_path}")
        print("="*70 + "\n")
        
        # Calculate basic statistics
        road_pixels = np.sum(segmentation_mask == 1)
        total_pixels = segmentation_mask.size
        road_percentage = (road_pixels / total_pixels) * 100
        
        print(f"📊 STATISTICS:")
        print(f"   Road pixels: {road_pixels:,}")
        print(f"   Total pixels: {total_pixels:,}")
        print(f"   Road coverage: {road_percentage:.2f}%")
        print("="*70 + "\n")
        
        print("💡 TIP: Check the overlay.png file to see the segmentation results!")
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up memory
        print("🧹 Cleaning up memory...")
        try:
            del model
            del image
            del image_batch
            del prediction
            gc.collect()
        except:
            pass

except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("\n💡 Install required packages:")
    print("   pip install tensorflow opencv-python numpy")
    sys.exit(1)

except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("🏁 Test completed safely!")