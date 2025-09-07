import os, pickle, joblib, hashlib, json, sys, traceback
from pathlib import Path

def _hash(path, algo="sha256", max_mb=16):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        remain = max_mb * 1024 * 1024
        while remain > 0:
            chunk = f.read(min(1024 * 1024, remain))
            if not chunk: break
            h.update(chunk)
            remain -= len(chunk)
    return f"{algo}:{h.hexdigest()} (first {max_mb}MB)"

def load_model_forgiving(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"model file not found: {path}")

    # Log file size + partial hash for sanity
    size_mb = p.stat().st_size / (1024*1024)
    print(f"[loader] loading {path} ({size_mb:.2f} MB), hash={_hash(path)}", flush=True)
    print(f"[loader] py={sys.version.split()[0]} "
          f"sklearn={_try_ver('sklearn')} joblib={_try_ver('joblib')} numpy={_try_ver('numpy')}", flush=True)

    # Try joblib (no mmap) → pickle → cloudpickle
    try:
        print(f"[loader] Attempting joblib.load for {path}", flush=True)
        return joblib.load(path, mmap_mode=None)
    except Exception as e1:
        print(f"[loader] joblib.load failed: {repr(e1)}", flush=True)
        print(f"[loader] Joblib error details: {str(e1)}", flush=True)
        
        try:
            print(f"[loader] Attempting pickle.load for {path}", flush=True)
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e2:
            print(f"[loader] pickle.load failed: {repr(e2)}", flush=True)
            print(f"[loader] Pickle error details: {str(e2)}", flush=True)
            
            try:
                print(f"[loader] Attempting cloudpickle.load for {path}", flush=True)
                import cloudpickle as cp
                with open(path, "rb") as f:
                    return cp.load(f)
            except Exception as e3:
                print(f"[loader] cloudpickle.load failed: {repr(e3)}", flush=True)
                print(f"[loader] Cloudpickle error details: {str(e3)}", flush=True)
                
                # Enhanced error message with more context
                error_msg = (
                    f"All loaders failed for {path}:\n"
                    f"  - joblib: {e1!r}\n"
                    f"  - pickle: {e2!r}\n"
                    f"  - cloudpickle: {e3!r}\n"
                    f"File size: {size_mb:.2f} MB\n"
                    f"File hash: {_hash(path)}\n"
                    f"Python: {sys.version}\n"
                    f"sklearn: {_try_ver('sklearn')}\n"
                    f"joblib: {_try_ver('joblib')}\n"
                    f"numpy: {_try_ver('numpy')}"
                )
                raise RuntimeError(error_msg)

def _try_ver(mod):
    try:
        m = __import__(mod)
        return getattr(m, "__version__", "unknown")
    except Exception:
        return "not-installed"


