"""
Simple TensorFlow Model Loader with Basic Compatibility
Handles common TensorFlow version issues.
"""

def try_load_model_simple(model_path):
    """
    Simple model loading with basic error handling
    """
    import tensorflow as tf
    
    print(f"📦 TensorFlow version: {tf.__version__}")
    print(f"🔄 Loading model: {model_path}")
    
    try:
        # Method 1: Standard loading
        model = tf.keras.models.load_model(model_path, compile=False)
        print("✅ Model loaded successfully")
        return model
        
    except Exception as e:
        print(f"❌ Error: {str(e)[:200]}...")
        
        if "groups" in str(e) and "Conv2DTranspose" in str(e):
            print("\n🔧 TensorFlow version compatibility issue detected!")
            print("💡 Your model was saved with a newer TensorFlow version.")
            print("\n📋 SOLUTIONS:")
            print("1. Update TensorFlow:")
            print("   conda install tensorflow=2.13")
            print("   # or")
            print("   pip install tensorflow==2.13")
            print("\n2. Or use a compatible version:")
            print("   pip install tensorflow==2.10")
            print("\n3. Or convert the model to a compatible format")
            
            return None
        else:
            print(f"\n❌ Unexpected error: {e}")
            return None

def get_model_input_output_size(model):
    """Get model input and output dimensions"""
    if model is None:
        return None, None
    
    input_shape = model.input_shape
    output_shape = model.output_shape
    
    print(f"📐 Model Input:  {input_shape}")
    print(f"📐 Model Output: {output_shape}")
    
    # Extract height, width from input shape
    if len(input_shape) >= 3:
        height, width = input_shape[1], input_shape[2]
        return (height, width), output_shape
    
    return input_shape, output_shape

if __name__ == "__main__":
    # Test the loader
    model_path = "models/RouteGenerator.h5"
    model = try_load_model_simple(model_path)
    
    if model:
        get_model_input_output_size(model)
    else:
        print("❌ Failed to load model")