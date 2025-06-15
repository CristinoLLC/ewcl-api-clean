import joblib
import numpy as np
import os

print("🚀 Starting model re-export process...")

try:
    print("📂 Loading ewcl_final_model.pkl...")
    final_model = joblib.load("models/ewcl_final_model.pkl")
    print("✅ Successfully loaded ewcl_final_model.pkl")
except Exception as e:
    print(f"❌ Failed to load ewcl_final_model.pkl: {e}")
    final_model = None

try:
    print("📂 Loading ewcl_regressor_v1.pkl...")
    regressor_model = joblib.load("models/ewcl_regressor_v1.pkl")
    print("✅ Successfully loaded ewcl_regressor_v1.pkl")
except Exception as e:
    print(f"❌ Failed to load ewcl_regressor_v1.pkl: {e}")
    regressor_model = None

# Re-export models that loaded successfully
if final_model is not None:
    try:
        print("💾 Re-exporting ewcl_final_model.pkl with protocol=2...")
        joblib.dump(final_model, "models/ewcl_final_model.pkl", protocol=2)
        print("✅ Successfully re-exported ewcl_final_model.pkl")
    except Exception as e:
        print(f"❌ Failed to re-export ewcl_final_model.pkl: {e}")

if regressor_model is not None:
    try:
        print("💾 Re-exporting ewcl_regressor_v1.pkl with protocol=2...")
        joblib.dump(regressor_model, "models/ewcl_regressor_v1.pkl", protocol=2)
        print("✅ Successfully re-exported ewcl_regressor_v1.pkl")
    except Exception as e:
        print(f"❌ Failed to re-export ewcl_regressor_v1.pkl: {e}")

print("🏁 Re-export process completed!")
