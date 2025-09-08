# MODEL_LOADING_GUIDE.md

## ⚠️ CRITICAL: EWCL Model Loading Guide

### The Problem
EWCL models are saved with **joblib**, NOT pickle! Using `pickle.load()` directly will fail with errors like:
```
_pickle.UnpicklingError: invalid load key, '\x10'
```

### The Solution
**ALWAYS** use the robust `load_model_forgiving()` function from `backend.models.loader`:

```python
from backend.models.loader import load_model_forgiving

# ✅ CORRECT - Use robust loader
model = load_model_forgiving(model_path)

# ❌ WRONG - Will fail with joblib models
with open(model_path, "rb") as f:
    model = pickle.load(f)
```

### Why This Happens
- Scikit-learn models are typically saved with `joblib.dump()` for better performance with NumPy arrays
- The `load_model_forgiving()` function tries multiple loading strategies:
  1. `joblib.load()` (primary - handles scikit-learn models)
  2. `pickle.load()` (fallback - handles pure Python objects)
  3. `cloudpickle.load()` (fallback - handles complex closures)

### Model Loading in All EWCL Routers
Every EWCL router should follow this pattern:

```python
from backend.models.loader import load_model_forgiving

def _load_model():
    """
    ⚠️  CRITICAL: EWCL models are saved with joblib, not pickle!
    Using pickle.load() directly will fail with "invalid load key" errors.
    Always use load_model_forgiving() which handles multiple formats.
    """
    path = os.environ.get("MODEL_PATH", "/app/models/model.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")
    
    # Use robust loader that handles joblib/pickle/cloudpickle automatically
    model = load_model_forgiving(path)
    return model
```

### Current Router Status
✅ **FIXED**: 
- `backend/api/routers/ewclv1.py` - Now uses `load_model_forgiving()`
- `backend/api/routers/ewclv1_M.py` - Uses `joblib.load()` directly
- `backend/api/routers/ewclv1p3.py` - Uses `load_model_forgiving()`
- `backend/api/routers/ewclv1_C.py` - Uses `joblib.load()` directly

### Model File Extensions
- Despite the `.pkl` extension, EWCL models are joblib format
- The extension is misleading but kept for compatibility
- Always assume `.pkl` files in the models/ directory are joblib format

### Debug Commands
To check model format:
```python
import joblib
import pickle

# Test joblib loading
try:
    model = joblib.load("model.pkl")
    print("✅ Model is joblib format")
except:
    print("❌ Model is NOT joblib format")

# Test pickle loading  
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    print("✅ Model is pickle format")
except:
    print("❌ Model is NOT pickle format")
```

### Future Development
- **NEW ROUTERS**: Always use `load_model_forgiving()`
- **MODEL TRAINING**: Save with `joblib.dump()` for consistency
- **DOCUMENTATION**: Include loading method in model metadata