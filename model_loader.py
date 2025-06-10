import joblib
import os

class EntropyCollapseModel:
    def __init__(self, model_path):
        """
        Unified entropy model that handles all EWCL prediction and disorder classification
        """
        self.model_path = model_path
        
        # Try to load the unified model, fallback to ewcl_model.pkl if not found
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            print(f"🧠 Unified Entropy Model loaded successfully from: {model_path}")
        elif os.path.exists("models/ewcl_model.pkl"):
            self.model = joblib.load("models/ewcl_model.pkl")
            print(f"⚠️ Unified model not found. Using fallback model: models/ewcl_model.pkl")
            self.model_path = "models/ewcl_model.pkl"
        else:
            raise FileNotFoundError(f"No model found at {model_path} or fallback location")

    def predict(self, features):
        """
        Predict using the loaded model
        """
        return self.model.predict(features)
    
    def predict_proba(self, features):
        """
        Predict probabilities if the model supports it
        """
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(features)
        else:
            return self.predict(features)
    
    def analyze(self, bf_series, plddt_series):
        """
        Analyze B-factor and pLDDT series for entropy/disorder prediction
        """
        if hasattr(self.model, 'analyze'):
            return self.model.analyze(bf_series, plddt_series)
        else:
            # Fallback to basic prediction using the synthetic model structure
            import pandas as pd
            import numpy as np
            
            # Convert series to features expected by the fallback model
            # The fallback model expects: ['score', 'avgEntropy', 'minEntropy', 'maxEntropy']
            score = np.mean(bf_series) if len(bf_series) > 0 else 50.0
            avg_entropy = np.mean(plddt_series) / 20.0 if len(plddt_series) > 0 else 2.5  # Scale pLDDT to entropy range
            min_entropy = np.min(plddt_series) / 20.0 if len(plddt_series) > 0 else 1.0
            max_entropy = np.max(plddt_series) / 20.0 if len(plddt_series) > 0 else 4.0
            
            features = np.array([[score, avg_entropy, min_entropy, max_entropy]])
            predictions = self.predict(features)
            
            return pd.DataFrame({
                'collapse_risk': predictions,
                'model_type': ['fallback'] * len(predictions)
            })