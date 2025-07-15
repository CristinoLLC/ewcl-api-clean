"""
Copy compatible models from Downloads to the models directory
"""

import shutil
from pathlib import Path
import os

def copy_models():
    """Copy compatible models from Downloads to models directory"""
    
    downloads = Path.home() / "Downloads"
    models_dir = Path(__file__).parent / "models"
    
    print(f"📂 Source: {downloads}")
    print(f"📂 Destination: {models_dir}")
    
    # Create models directory if it doesn't exist
    models_dir.mkdir(exist_ok=True)
    
    # Look for DisProt and hallucination models in Downloads
    model_searches = [
        # DisProt models
        ("EWCL_20K_Benchmark/models/hallucination_classifier_v3_api.pkl", "xgb_disprot_model.pkl"),
        ("EWCL_20K_Benchmark/models/hallucination_classifier_v3.pkl", "hallucination_detector.pkl"),
        ("EWCL_FullBackup/ewcl_ai_models/hallucination_safe_model_v3000.pkl", "hallucination_detector_v3000.pkl"),
        ("EWCL_FullBackup/ewcl_ai_models/hallucination_safe_model_v500.pkl", "hallucination_detector_v500.pkl"),
        
        # Additional regressor models
        ("ewcl_regressor_model (1).pkl", "ewcl_regressor_model_v2.pkl"),
        ("ewcl_regressor_af.pkl", "ewcl_regressor_af.pkl"),
        ("ewcl_ai_model.pkl", "ewcl_ai_model.pkl"),
        ("ewcl_regressor_v1.pkl", "ewcl_regressor_v1.pkl"),
        
        # High-confidence models
        ("ewcl_residue_local_highmid_model.pkl", "ewcl_residue_local_high_model_v2.pkl"),
        ("ewcl_residue_local_highmid_scaler.pkl", "ewcl_residue_local_high_scaler_v2.pkl"),
    ]
    
    copied = 0
    found_models = []
    
    print("\n🔍 Searching for models...")
    
    for search_path, dest_name in model_searches:
        source = downloads / search_path
        dest = models_dir / dest_name
        
        if source.exists():
            try:
                shutil.copy2(source, dest)
                size = dest.stat().st_size
                print(f"✅ Copied: {search_path} -> {dest_name} ({size} bytes)")
                found_models.append(dest_name)
                copied += 1
            except Exception as e:
                print(f"❌ Failed to copy {search_path}: {e}")
        else:
            print(f"⚠️ Not found: {search_path}")
    
    # Try to find any pkl files in common locations
    common_locations = [
        downloads / "EWCL_20K_Benchmark" / "models",
        downloads / "EWCL_FullBackup" / "ewcl_ai_models",
        downloads,
    ]
    
    print(f"\n🔍 Scanning for additional .pkl files...")
    
    for location in common_locations:
        if location.exists():
            pkl_files = list(location.glob("*.pkl"))
            if pkl_files:
                print(f"📋 Found {len(pkl_files)} .pkl files in {location}:")
                for pkl in pkl_files[:5]:  # Show first 5
                    print(f"  • {pkl.name}")
                if len(pkl_files) > 5:
                    print(f"  ... and {len(pkl_files) - 5} more")
    
    print(f"\n📊 Summary:")
    print(f"  • {copied} models copied successfully")
    print(f"  • Models available: {found_models}")
    
    # List all files in models directory
    print(f"\n📂 Current models directory contents:")
    for model_file in sorted(models_dir.glob("*.pkl")):
        size = model_file.stat().st_size
        print(f"  • {model_file.name} ({size:,} bytes)")
    
    return copied > 0

if __name__ == "__main__":
    success = copy_models()
    if success:
        print("\n✅ Models copied! Restart your API to load them.")
    else:
        print("\n⚠️ No models found to copy. Check your Downloads folder.")
