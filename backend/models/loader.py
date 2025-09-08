from __future__ import annotations
import os, pickle, joblib, hashlib, json, sys, traceback, io, re
from pathlib import Path

LFS_PREFIX = b"version https://git-lfs.github.com/spec/v1"

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
    """
    Try joblib -> cloudpickle -> pickle, all in-memory.
    Raises RuntimeError("All loaders failed: ...") if none succeed.
    """
    import joblib, pickle
    try:
        import cloudpickle  # ensure import error surfaces early if not installed
    except Exception:
        cloudpickle = None

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"model file not found: {path}")

    # Log file size + partial hash for sanity
    size_mb = p.stat().st_size / (1024*1024)
    print(f"[loader] loading {path} ({size_mb:.2f} MB), hash={_hash(path)}", flush=True)
    print(f"[loader] py={sys.version.split()[0]} "
          f"sklearn={_try_ver('sklearn')} joblib={_try_ver('joblib')} numpy={_try_ver('numpy')}", flush=True)

    with open(path, "rb") as f:
        blob = f.read()

    # --- Guard: LFS pointer or HTML error page? ---
    head = blob[:200].lower()
    if head.startswith(LFS_PREFIX):
        raise RuntimeError(
            "Model file is a Git LFS pointer, not the binary. Run `git lfs pull` "
            "or provision the real artifact at this path."
        )
    if head.startswith(b"<!doctype html") or b"<html" in head:
        raise RuntimeError("Model path contains an HTML page, not a model (check your download).")

    last_err = None
    # 1) joblib
    try:
        print(f"[loader] Attempting joblib.load for {path}", flush=True)
        return joblib.load(io.BytesIO(blob))
    except Exception as e:
        print(f"[loader] joblib.load failed: {repr(e)}", flush=True)
        last_err = e

    # 2) cloudpickle
    if cloudpickle is not None:
        try:
            print(f"[loader] Attempting cloudpickle.loads for {path}", flush=True)
            return cloudpickle.loads(blob)
        except Exception as e:
            print(f"[loader] cloudpickle.loads failed: {repr(e)}", flush=True)
            last_err = e

    # 3) stdlib pickle
    try:
        print(f"[loader] Attempting pickle.loads for {path}", flush=True)
        return pickle.loads(blob)
    except Exception as e:
        print(f"[loader] pickle.loads failed: {repr(e)}", flush=True)
        last_err = e

    raise RuntimeError(f"All loaders failed: {last_err}")

def _try_ver(mod):
    try:
        m = __import__(mod)
        return getattr(m, "__version__", "unknown")
    except Exception:
        return "not-installed"


