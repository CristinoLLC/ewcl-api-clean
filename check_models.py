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

"""
Check for required model files and provide download instructions
"""

import os
from pathlib import Path

def check_models():
    """Check if all required models are present"""
    
    models_dir = Path(__file__).parent / "models"
    
    required_models = {
        "ewcl_regressor_model.pkl": "Main EWCL regressor model",
        "ewcl_residue_local_high_model.pkl": "High-confidence refiner model", 
        "ewcl_residue_local_high_scaler.pkl": "High-confidence scaler",
        "hallucination_detector_model.pkl": "Hallucination detector",
        "xgb_disprot_model.pkl": "DisProt disorder prediction model",
        "hallucination_detector.pkl": "DisProt hallucination detector"
    }
    
    print("🔍 Checking for required model files...")
    print(f"📂 Models directory: {models_dir}")
    
    if not models_dir.exists():
        print("❌ Models directory does not exist!")
        return False
    
    missing_models = []
    present_models = []
    
    for model_file, description in required_models.items():
        model_path = models_dir / model_file
        if model_path.exists():
            size = model_path.stat().st_size
            print(f"✅ {model_file} ({size} bytes) - {description}")
            present_models.append(model_file)
        else:
            print(f"❌ {model_file} - {description}")
            missing_models.append(model_file)
    
    print(f"\n📊 Summary: {len(present_models)}/{len(required_models)} models found")
    
    if missing_models:
        print("\n⚠️ Missing models:")
        for model in missing_models:
            print(f"  • {model}")
        
        print("\n🔧 To add missing models:")
        print("1. Download the model files")
        print("2. Place them in the models/ directory")
        print("3. Restart the API")
        
        return False
    else:
        print("\n✅ All required models are present!")
        return True

if __name__ == "__main__":
    check_models()
