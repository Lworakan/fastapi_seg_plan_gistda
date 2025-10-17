"""
Model Diagnostic Script
Simple script to diagnose model loading issues without complex dependencies.
"""

import os
import sys

def diagnose_model_issue():
    """Diagnose the model loading issue"""
    
    print("="*70)
    print("🔍 MODEL LOADING DIAGNOSTICS")
    print("="*70)
    
    # Check TensorFlow
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow found: {tf.__version__}")
        
        # Check model file
        model_path = "models/RouteGenerator.h5"
        if os.path.exists(model_path):
            print(f"✅ Model file exists: {model_path}")
            
            # Get file size
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"📏 Model size: {size_mb:.1f} MB")
            
            # Try to inspect the model without loading
            try:
                # Check HDF5 structure
                import h5py
                with h5py.File(model_path, 'r') as f:
                    print("✅ HDF5 file structure is valid")
                    
                    # Look for model config
                    if 'model_config' in f.attrs:
                        print("✅ Model config found in file")
                    else:
                        print("⚠️  No model config in file attributes")
                        
            except ImportError:
                print("⚠️  h5py not available for detailed inspection")
            except Exception as e:
                print(f"❌ HDF5 structure issue: {e}")
                
        else:
            print(f"❌ Model file not found: {model_path}")
            return
            
        # Try basic model loading
        print("\n🔄 Attempting to load model...")
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            print("✅ Model loaded successfully!")
            print(f"📐 Input shape: {model.input_shape}")
            print(f"📐 Output shape: {model.output_shape}")
            print(f"🔢 Parameters: {model.count_params():,}")
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Model loading failed:")
            print(f"   {error_msg[:200]}...")
            
            # Analyze the error
            if "groups" in error_msg and "Conv2DTranspose" in error_msg:
                print("\n💡 DIAGNOSIS: TensorFlow Version Compatibility Issue")
                print("   Your model was saved with TensorFlow >= 2.12")
                print(f"   But you're using TensorFlow {tf.__version__}")
                print("\n🔧 SOLUTIONS:")
                print("   1. Update TensorFlow:")
                print("      conda install tensorflow=2.13")
                print("   2. Or use TensorFlow 2.12+:")
                print("      pip install tensorflow>=2.12")
                
            elif "Unknown layer" in error_msg:
                print("\n💡 DIAGNOSIS: Custom Layer Issue")
                print("   Model contains custom layers not available")
                
            elif "format" in error_msg.lower():
                print("\n💡 DIAGNOSIS: Model Format Issue")
                print("   Model file may be corrupted or incompatible")
                
            else:
                print(f"\n💡 DIAGNOSIS: Unknown Issue")
                print(f"   Error: {error_msg}")
            
    except ImportError:
        print("❌ TensorFlow not found")
        print("   Install with: pip install tensorflow")
        return
        
    print("\n" + "="*70)
    print("📋 SUMMARY")
    print("="*70)
    print("The most likely issue is TensorFlow version compatibility.")
    print("Try updating TensorFlow to version 2.12 or higher.")
    print("="*70 + "\n")

if __name__ == "__main__":
    diagnose_model_issue()