"""
Script to copy compatible models from Downloads to the models directory
"""

import shutil
from pathlib import Path
import os

def copy_models():
    """Copy compatible models from Downloads"""
    
    downloads = Path.home() / "Downloads"
    models_dir = Path(__file__).parent / "models"
    
    print(f"📂 Models directory: {models_dir}")
    print(f"📥 Downloads directory: {downloads}")
    
    # Create models directory if it doesn't exist
    models_dir.mkdir(exist_ok=True)
    
    # Model mapping: source -> destination
    model_mappings = {
        # Try different hallucination models
        "EWCL_20K_Benchmark/models/hallucination_classifier_v3_api.pkl": "hallucination_detector.pkl",
        "EWCL_20K_Benchmark/models/hallucination_classifier_v3.pkl": "hallucination_detector_backup.pkl",
        "EWCL_FullBackup/ewcl_ai_models/hallucination_safe_model_v3000.pkl": "hallucination_detector_v3000.pkl",
        
        # Try different regressor models
        "ewcl_regressor_model (1).pkl": "ewcl_regressor_model_backup.pkl",
        "ewcl_regressor_af.pkl": "ewcl_regressor_af.pkl",
        "ewcl_ai_model.pkl": "ewcl_ai_model.pkl",
        
        # Try different high-confidence models
        "ewcl_residue_local_highmid_model.pkl": "ewcl_residue_local_high_model_backup.pkl",
        "ewcl_residue_local_highmid_scaler.pkl": "ewcl_residue_local_high_scaler_backup.pkl",
    }
    
    copied_count = 0
    
    for source_path, dest_name in model_mappings.items():
        source = downloads / source_path
        dest = models_dir / dest_name
        
        if source.exists():
            try:
                shutil.copy2(source, dest)
                print(f"✅ Copied {source_path} -> {dest_name}")
                print(f"   Size: {dest.stat().st_size} bytes")
                copied_count += 1
            except Exception as e:
                print(f"❌ Failed to copy {source_path}: {e}")
        else:
            print(f"⚠️ Not found: {source_path}")
    
    print(f"\n📊 Summary: {copied_count} models copied")
    
    # List all models in the models directory
    print(f"\n📋 Models in {models_dir}:")
    for model_file in sorted(models_dir.glob("*.pkl")):
        size = model_file.stat().st_size
        print(f"  • {model_file.name} ({size} bytes)")

if __name__ == "__main__":
    copy_models()
