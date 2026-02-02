import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Import our custom modules
from src.data_processor import download_and_process_data, create_windows
from src.quantum_reservoir import QuantumReservoir

def main():
    # --- CONFIGURATION ---
    TICKER = 'SPY'
    WINDOW_SIZE = 5
    SCALE_FACTOR = 80.0  # Tuned parameter from research
    RIDGE_ALPHA = 0.01   # Tuned parameter from research
    
    # 1. Load Data
    data = download_and_process_data(TICKER)
    X_raw, y = create_windows(data['Log_Returns'], WINDOW_SIZE)
    
    # 2. Split Data (80% Train, 20% Test)
    split_point = int(len(X_raw) * 0.8)
    X_train_raw = X_raw[:split_point]
    X_test_raw = X_raw[split_point:]
    y_train = y[:split_point]
    y_test = y[split_point:]
    
    print(f"Training set: {len(X_train_raw)} samples")
    print(f"Testing set:  {len(X_test_raw)} samples")
    
    # 3. Initialize Quantum Reservoir
    qrc = QuantumReservoir(n_qubits=WINDOW_SIZE, scale_factor=SCALE_FACTOR)
    
    # 4. Quantum Feature Extraction Loop
    print("\n--- STARTING QUANTUM SIMULATION ---")
    
    print("Processing Training Data...")
    X_train_quantum = []
    for window in tqdm(X_train_raw):
        features = qrc.get_features(window)
        X_train_quantum.append(features)
        
    print("Processing Testing Data...")
    X_test_quantum = []
    for window in tqdm(X_test_raw):
        features = qrc.get_features(window)
        X_test_quantum.append(features)
        
    # 5. Train Readout Layer (Ridge Regression)
    print("\n--- TRAINING READOUT LAYER ---")
    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(X_train_quantum, y_train)
    
    # 6. Evaluate
    predictions = model.predict(X_test_quantum)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    print(f"\n==============================")
    print(f"FINAL RESULTS")
    print(f"==============================")
    print(f"Model: QRC (Scale={SCALE_FACTOR})")
    print(f"RMSE:  {rmse:.5f}")
    
    # 7. Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(y_test[:50], label='Actual Volatility (Proxy)', color='gray', alpha=0.6)
    plt.plot(predictions[:50], label='Quantum Prediction', color='red', linewidth=2)
    plt.title(f'Quantum Reservoir Forecasting (RMSE: {rmse:.5f})')
    plt.xlabel('Days (Test Set)')
    plt.ylabel('Volatility Magnitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig("volatility_forecast.png")
    print("Graph saved as 'volatility_forecast.png'")
    plt.show()

if __name__ == "__main__":
    main()