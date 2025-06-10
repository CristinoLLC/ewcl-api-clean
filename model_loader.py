import joblib

class EntropyCollapseModel:
    def __init__(self, model_path):
        """
        Unified entropy model that handles all EWCL prediction and disorder classification
        """
        self.model = joblib.load(model_path)
        self.model_path = model_path
        print(f"🧠 Unified Entropy Model loaded successfully from: {model_path}")

    def predict(self, features):
        """
        Predict using the unified model
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
            # Fallback to basic prediction if analyze method doesn't exist
            import pandas as pd
            features = pd.concat([bf_series, plddt_series], axis=1)
            predictions = self.predict(features)
            return pd.DataFrame({'predictions': predictions})