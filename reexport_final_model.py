import joblib
import numpy as np
import os

print("🚀 Starting final model re-export process...")
print(f"📊 NumPy version: {np.__version__}")

# Check if file exists
model_path = "models/ewcl_final_model.pkl"
if os.path.exists(model_path):
    print(f"✅ Found model file: {model_path}")
    file_size = os.path.getsize(model_path)
    print(f"📁 File size: {file_size} bytes")
else:
    print(f"❌ Model file not found: {model_path}")
    exit(1)

try:
    print("🔄 Loading model...")
    model = joblib.load(model_path)
    print("✅ Model loaded successfully")
    print(f"📋 Model type: {type(model)}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit(1)

try:
    print("💾 Re-exporting with protocol=2...")
    joblib.dump(model, model_path, protocol=2)
    print("✅ Re-exported ewcl_final_model.pkl with protocol=2")
    
    # Verify the re-export
    print("🔍 Verifying re-exported model...")
    test_model = joblib.load(model_path)
    print("✅ Verification successful")
    
    # Provide feedback on numpy version compatibility
    print("✅ NumPy version:", np.__version__)
    
except Exception as e:
    print(f"❌ Failed to re-export model: {e}")
    exit(1)

print("🏁 Re-export process completed successfully!")
