import pandas as pd
import numpy as np
import os
import pickle
import sys

# --- PATH CONFIGURATION ---
root_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(root_dir, 'data', 'diabetes.csv')
MODEL_PATH = os.path.join(root_dir, 'models', 'diabetes_model.pkl')
SCALER_PATH = os.path.join(root_dir, 'models', 'scaler.pkl')


def generate_smart_data(filepath):
    """Generates synthetic data with medical correlations."""
    print("[INFO] Generating SMART Medical Data with SAFETY NET...")

    np.random.seed(42)
    n_samples = 2000

    # 1. Generate Realistic Patient Features
    data = {
        'Pregnancies': np.random.randint(0, 15, n_samples),
        'Glucose': np.random.randint(60, 200, n_samples),
        'BloodPressure': np.random.randint(50, 120, n_samples),
        'SkinThickness': np.random.randint(0, 60, n_samples),
        'Insulin': np.random.randint(0, 600, n_samples),
        'BMI': np.random.uniform(18.0, 60.0, n_samples).round(1),
        'DiabetesPedigreeFunction': np.random.uniform(0.08, 2.0, n_samples).round(3),
        'Age': np.random.randint(21, 85, n_samples)
    }
    df = pd.DataFrame(data)

    # 2. Risk Calculation (The Logic)
    score = (
            (df['Glucose'] - 100) * 0.6 +
            (df['BMI'] - 25) * 0.4 +
            (df['Age'] - 30) * 0.1 +
            (df['DiabetesPedigreeFunction'] * 5)
    )

    # 3. Assign Outcomes
    score = (score - score.mean()) / score.std()
    probability = 1 / (1 + np.exp(-score))
    df['Outcome'] = (probability > 0.5).astype(int)

    # 4. Safety Net Overrides
    # Athlete Rule
    mask_healthy = (df['Glucose'] < 105) & (df['BMI'] < 26)
    df.loc[mask_healthy, 'Outcome'] = 0

    # High Risk Rule
    mask_sick = (df['Glucose'] > 155)
    df.loc[mask_sick, 'Outcome'] = 1

    # 5. Save
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"[SUCCESS] Smart dataset saved to: {filepath}")
    return df


def train_model():
    print(f"[INFO] Working Directory: {root_dir}")

    # Always generate fresh data for consistent results
    df = generate_smart_data(DATA_PATH)

    try:
        from sklearn.model_selection import train_test_split
        from sklearn.neural_network import MLPClassifier  # <--- NEW: Neural Network
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score

        X = df.drop('Outcome', axis=1)
        y = df['Outcome']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scaling is CRITICAL for Neural Networks
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print("[INFO] Training Neural Network (MLP)...")

        # --- NEURAL NETWORK ARCHITECTURE ---
        # Hidden Layers: (16, 8) -> Two layers with 16 and 8 neurons respectively
        # Activation: 'relu' (Standard for deep learning)
        # Max Iterations: 1000 (To allow enough time to converge)
        model = MLPClassifier(hidden_layer_sizes=(16, 8),
                              activation='relu',
                              solver='adam',
                              max_iter=1000,
                              random_state=42)

        model.fit(X_train_scaled, y_train)

        acc = accuracy_score(y_test, model.predict(X_test_scaled))
        print(f"[INFO] Neural Network Accuracy: {acc:.2%}")

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(scaler, f)

        print("[SUCCESS] Training Complete & Saved!")

    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    train_model()