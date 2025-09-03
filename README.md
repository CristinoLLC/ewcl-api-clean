# EWCL Physics + Proxy API

Model-free backend focused on physics-based EWCL and a proxy EWCL (entropy-weighted pLDDT/B). No ML models required.

## Features

- Physics-only EWCL: entropy/structural signal analysis
- EWCL-Proxy: re-interpretation of pLDDT/B with local entropy
- Reverse EWCL: instability highlighting via rev_cl = 1 - cl

## API Endpoints

- `POST /api/analyze/raw` - Physics-only EWCL (CA-only)
- `POST /api/analyze/proxy` - EWCL-Proxy (CA-only)
- `POST /analyze-rev-ewcl` - Reverse EWCL for instability analysis
- `POST /analyze-pdb` - Unified physics endpoint (back-compat)

## Deployment

### Local Development
```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Render Deployment
1. Push to GitHub repository
2. Connect to Render
3. Deploy with `render.yaml` configuration

### Railway Deployment
Railway deployment with automatic Git LFS model downloads and router fixes applied.

No ML model files required.
