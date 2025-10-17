"""
Model Compatibility Helper
Handles TensorFlow version compatibility issues when loading models.
"""

import tensorflow as tf
from tensorflow.keras.models import load_model
import warnings

def load_model_with_compatibility(model_path, compile=False):
    """
    Load model with automatic compatibility handling for different TensorFlow versions.
    
    Args:
        model_path: Path to the .h5 model file
        compile: Whether to compile the model
    
    Returns:
        Loaded Keras model
    """
    print(f"🔄 Attempting to load model: {model_path}")
    
    # First, try normal loading
    try:
        model = load_model(model_path, compile=compile)
        print("✅ Model loaded successfully (standard method)")
        return model
    except Exception as e:
        print(f"⚠️  Standard loading failed: {str(e)[:100]}...")
    
    # If that fails, try with custom objects to handle compatibility issues
    print("🔧 Trying compatibility mode...")
    
    try:
        # Create custom objects to handle known compatibility issues
        custom_objects = {}
        
        # Handle Conv2DTranspose 'groups' parameter issue
        def create_conv2d_transpose(**config):
            # Remove unsupported parameters
            if 'groups' in config:
                del config['groups']
            return tf.keras.layers.Conv2DTranspose.from_config(config)
        
        # Add more compatibility handlers as needed
        custom_objects['Conv2DTranspose'] = tf.keras.layers.Conv2DTranspose
        
        # Try loading with warnings suppressed
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = load_model(model_path, compile=compile, custom_objects=custom_objects)
        
        print("✅ Model loaded successfully (compatibility mode)")
        return model
        
    except Exception as e2:
        print(f"❌ Compatibility mode also failed: {str(e2)[:100]}...")
        
        # Last resort: try to load just the weights
        print("🔧 Attempting weights-only loading...")
        try:
            # This is a more complex approach that would require model reconstruction
            print("❌ Weights-only loading not implemented yet")
            raise e2
        except:
            print("\n💡 TROUBLESHOOTING TIPS:")
            print("   1. Check TensorFlow version compatibility:")
            print(f"      Current version: {tf.__version__}")
            print("   2. Try updating TensorFlow:")
            print("      pip install tensorflow>=2.12")
            print("   3. Or downgrade to match the model's TF version")
            print("   4. Re-save the model with current TensorFlow version")
            raise e2

def get_model_info(model):
    """Get detailed information about the loaded model"""
    info = {
        'input_shape': model.input_shape,
        'output_shape': model.output_shape,
        'total_params': model.count_params(),
        'trainable_params': sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]),
        'non_trainable_params': sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    }
    return info

def print_model_summary(model):
    """Print a nice summary of the model"""
    info = get_model_info(model)
    
    print(f"\n{'='*50}")
    print("📊 MODEL INFORMATION")
    print("="*50)
    print(f"📐 Input shape:  {info['input_shape']}")
    print(f"📐 Output shape: {info['output_shape']}")
    print(f"🔢 Total parameters: {info['total_params']:,}")
    print(f"🎯 Trainable: {info['trainable_params']:,}")
    print(f"🔒 Non-trainable: {info['non_trainable_params']:,}")
    print("="*50 + "\n")