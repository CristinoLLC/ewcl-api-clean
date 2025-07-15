# EWCL Collapse-Likelihood API

Physics-based + ML refined EWCL with hallucination flags and reverse collapse likelihood analysis.

## Features

- **Physics-only EWCL**: Pure entropy/structural signal analysis
- **ML-enhanced predictions**: Regressor and refinement models (with fallbacks)
- **Hallucination detection**: Identifies potentially unreliable predictions
- **Reverse EWCL**: Entropy-based collapse likelihood inversion for instability analysis
- **DisProt prediction**: Disorder prediction with physics-based fallback

## API Endpoints

- `POST /analyze-pdb` - Unified analysis with complete physics features
- `POST /analyze-rev-ewcl` - Reverse EWCL for instability analysis
- `POST /disprot-predict` - DisProt disorder prediction
- `POST /api/analyze/raw` - Physics-only EWCL
- `POST /api/analyze/regressor` - Physics + ML regressor
- `POST /api/analyze/refined` - Physics + refined model
- `POST /api/analyze/hallucination` - Physics + hallucination detection

## Deployment

### Local Development
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Render Deployment
1. Push to GitHub repository
2. Connect to Render
3. Deploy with `render.yaml` configuration

## Model Files

Place ML model files in the `models/` directory:
- `ewcl_regressor_model.pkl`
- `ewcl_residue_local_high_model.pkl`
- `ewcl_residue_local_high_scaler.pkl`
- `hallucination_detector.pkl`

The API works with physics-based analysis even if ML models are missing.
