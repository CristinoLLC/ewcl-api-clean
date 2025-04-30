import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle

# Create directory for models if it doesn't exist
os.makedirs('models', exist_ok=True)

# Generate synthetic data for training
# In a real scenario, you would load your data from a file or database
def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    score = np.random.uniform(0, 100, n_samples)
    avg_entropy = np.random.uniform(0, 5, n_samples)
    min_entropy = np.random.uniform(0, avg_entropy)
    max_entropy = np.random.uniform(avg_entropy, 10, n_samples)
    
    # Create a synthetic target based on features
    collapse_risk = 0.2 * score + 0.3 * avg_entropy - 0.1 * min_entropy + 0.4 * max_entropy
    collapse_risk = collapse_risk / collapse_risk.max()  # Normalize to [0,1]
    
    # Add some noise
    collapse_risk = collapse_risk + np.random.normal(0, 0.1, n_samples)
    collapse_risk = np.clip(collapse_risk, 0, 1)  # Ensure values stay in [0,1]
    
    # Create a dataframe
    data = pd.DataFrame({
        'score': score,
        'avgEntropy': avg_entropy,
        'minEntropy': min_entropy,
        'maxEntropy': max_entropy,
        'collapseRisk': collapse_risk
    })
    
    return data

if __name__ == "__main__":
    print("Generating synthetic data for training...")
    data = generate_synthetic_data()
    
    # Split features and target
    X = data[['score', 'avgEntropy', 'minEntropy', 'maxEntropy']]
    y = data['collapseRisk']
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training the model...")
    # Train a Random Forest regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model MSE: {mse:.4f}")
    
    # Save the model
    model_path = os.path.join("models", "ewcl_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    print(f"✅ Model saved successfully to {model_path}")
    
    # Test prediction with sample data
    test_sample = np.array([[50, 2.5, 1.2, 4.0]])
    prediction = model.predict(test_sample)[0]
    print(f"Sample prediction (collapseRisk) for {test_sample}: {prediction:.4f}")