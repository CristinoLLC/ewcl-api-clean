# EWCL Backend Model Bundle

This bundle contains both EWCL disorder prediction models ready for production deployment.

## Models Included

### EWCLv1-M (Recommended)
- **File**: `models/EWCLv1-M.pkl`
- **Type**: LightGBM Classifier
- **Features**: 255 sequence-based features
- **Performance**: Optimized with 1515 boosting iterations
- **Description**: Machine learning optimized version with enhanced feature engineering

### EWCLv1 (Classic)
- **File**: `models/EWCLv1.pkl`  
- **Type**: LightGBM Classifier
- **Features**: 249 sequence-based features
- **Description**: Original EWCL model for backward compatibility

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements-backend.txt
   ```

2. **Load models**:
   ```python
   import joblib
   
   # Load EWCLv1-M (recommended)
   model_m = joblib.load('models/EWCLv1-M.pkl')
   
   # Load EWCLv1 (classic)
   model_classic = joblib.load('models/EWCLv1.pkl')
   ```

3. **Feature extraction**:
   ```python
   from meta.ewcl_feature_extractor_v2 import EWCLFeatureExtractor
   
   extractor = EWCLFeatureExtractor()
   features = extractor.extract_sequence_features(sequence, protein_id)
   ```

## Environment

- **Python**: 3.11.8
- **Platform**: macOS ARM64 (compatible with Linux x86_64)
- **Created**: 2025-08-31T16:37:03Z

## Files

```
ewcl_backend_bundle/
├── models/
│   ├── EWCLv1-M.pkl           # LightGBM model (255 features)
│   └── EWCLv1.pkl             # LightGBM model (249 features)
├── meta/
│   ├── EWCLv1-M_feature_info.json
│   ├── EWCLv1_feature_info.json
│   └── ewcl_feature_extractor_v2.py
├── models_manifest.json       # Complete model metadata
├── requirements-backend.txt   # Pinned dependencies
└── README.md                  # This file
```

## Checksums

- **EWCLv1-M.pkl**: 4d0cfbeccfdae55c4b57f5fe35943b76ac0502fc4a28e4d113f6c60d905b257f
- **EWCLv1.pkl**: 08c48a7a4f980acb3f2a2813b2bb2150ecab0dec6b25be16605f1b9e6ec2d348
- **ewcl_feature_extractor_v2.py**: e395d6aaecb78149a2a2e370581cc49af8163c0e805879994ea16d6cfb2919c9

## API Integration

Both models are designed for REST API deployment with FastAPI. See `models_manifest.json` for complete deployment specifications.
