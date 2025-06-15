import os

model_path = "models/ewcl_final_model.pkl"

print("🔍 Checking model file...")

if os.path.exists(model_path):
    file_size = os.path.getsize(model_path)
    print(f"✅ File exists: {model_path}")
    print(f"📊 File size: {file_size} bytes")
    
    # Check first 50 bytes
    with open(model_path, "rb") as f:
        first_bytes = f.read(50)
    print(f"🔍 First 50 bytes: {first_bytes}")
    
    # Check if it starts with newline (corruption indicator)
    if first_bytes.startswith(b'\n') or b'\x0a' in first_bytes[:5]:
        print("❌ Model file is corrupted (contains newlines at start)")
    else:
        print("✅ Model file looks like proper binary")
        
    # Try to load with different methods
    try:
        import joblib
        model = joblib.load(model_path)
        print("✅ Model loads successfully with joblib")
        print(f"📋 Model type: {type(model)}")
    except Exception as e:
        print(f"❌ Joblib loading failed: {e}")
        
    try:
        import pickle
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print("✅ Model loads successfully with pickle")
    except Exception as e:
        print(f"❌ Pickle loading failed: {e}")
        
else:
    print(f"❌ File does not exist: {model_path}")
