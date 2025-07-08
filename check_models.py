"""
Script to validate model files before deployment
"""

from pathlib import Path
import joblib

MODEL_DIR = Path("models")

def check_model_file(model_path, model_name):
    """Check if model file exists and can be loaded"""
    print(f"\n🔍 Checking {model_name}:")
    print(f"  📁 Path: {model_path}")
    print(f"  📊 Exists: {model_path.exists()}")
    
    if model_path.exists():
        print(f"  📏 Size: {model_path.stat().st_size:,} bytes")
        try:
            model = joblib.load(model_path)
            print(f"  ✅ Loads successfully")
            print(f"  🏷️  Type: {type(model).__name__}")
            return True
        except Exception as e:
            print(f"  ❌ Load failed: {e}")
            return False
    else:
        print(f"  ❌ File not found")
        return False

if __name__ == "__main__":
    print("🔬 EWCL Model Validation")
    print(f"📂 Model directory: {MODEL_DIR.absolute()}")
    
    models = [
        ("ewcl_regressor_model.pkl", "Main Regressor"),
        ("ewcl_residue_local_high_model.pkl", "High Refiner"),
        ("ewcl_residue_local_high_scaler.pkl", "Scaler"),
        ("hallucination_detector_model.pkl", "Hallucination Detector"),
    ]
    
    results = []
    for filename, name in models:
        success = check_model_file(MODEL_DIR / filename, name)
        results.append((name, success))
    
    print(f"\n📊 Summary:")
    for name, success in results:
        status = "✅" if success else "❌"
        print(f"  {status} {name}")
    
    all_good = all(success for _, success in results)
    print(f"\n🎯 Overall: {'All models ready!' if all_good else 'Some models missing/broken'}")
