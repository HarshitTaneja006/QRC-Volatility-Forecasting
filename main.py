import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Import our custom modules
from src.data_processor import download_and_process_data, create_windows
from src.macro_processor import download_macro_data
from src.quantum_reservoir import QuantumReservoir

def main():
    # --- CONFIGURATION ---
    TICKER = 'SPY'
    WINDOW_SIZE = 5
    SCALE_FACTOR = 80.0  # Tuned parameter from research
    RIDGE_ALPHA = 0.01   # Tuned parameter from research
    
    # ==========================================
    # EXPERIMENT 1: S&P 500 (FINANCIAL)
    # ==========================================
    print("\n==========================================")
    print("RUNNING EXPERIMENT 1: S&P 500 VOLATILITY")
    print("==========================================")
    
    # 1. Load Data
    data = download_and_process_data(TICKER)
    X_raw, y = create_windows(data['Log_Returns'], WINDOW_SIZE)
    
    # 2. Split Data (80% Train, 20% Test)
    split_point = int(len(X_raw) * 0.8)
    X_train_raw = X_raw[:split_point]
    X_test_raw = X_raw[split_point:]
    y_train = y[:split_point]
    y_test = y[split_point:]
    
    # 3. Initialize Quantum Reservoir
    qrc = QuantumReservoir(n_qubits=WINDOW_SIZE, scale_factor=SCALE_FACTOR)
    
    # 4. Quantum Feature Extraction Loop
    print("\n--- STARTING QUANTUM SIMULATION (SPY) ---")
    
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
        
    # 5. Train Readout Layer
    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(X_train_quantum, y_train)
    predictions = model.predict(X_test_quantum)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    print(f"S&P 500 RMSE: {rmse:.5f}")
    
    # Save Figure 3
    plt.figure(figsize=(12, 6))
    plt.plot(y_test[:50], label='Actual Volatility', color='gray', alpha=0.6)
    plt.plot(predictions[:50], label='Quantum Prediction', color='red', linewidth=2)
    plt.title(f'S&P 500 Forecast (RMSE: {rmse:.5f})')
    plt.legend()
    plt.savefig("volatility_forecast_spy.png")
    print("Graph saved as 'volatility_forecast_spy.png'")

    # ==========================================
    # EXPERIMENT 2: GDP (MACROECONOMIC)
    # ==========================================
    print("\n==========================================")
    print("RUNNING EXPERIMENT 2: GDP VOLATILITY")
    print("==========================================")
    
    # 1. Download & Prepare Data
    gdp_data, _ = download_macro_data()
    
    if gdp_data is not None:
        X_gdp, y_gdp = create_windows(gdp_data, WINDOW_SIZE)
        
        # 2. Split Data
        split_gdp = int(len(X_gdp) * 0.8)
        X_train_gdp = X_gdp[:split_gdp]
        X_test_gdp = X_gdp[split_gdp:]
        y_train_gdp = y_gdp[:split_gdp]
        y_test_gdp = y_gdp[split_gdp:]
        
        # 3. Run Quantum Simulation (Reuse QRC instance)
        print("\n--- STARTING QUANTUM SIMULATION (GDP) ---")
        
        print("Processing GDP Training Data...")
        X_train_gdp_q = []
        for window in tqdm(X_train_gdp):
            features = qrc.get_features(window)
            X_train_gdp_q.append(features)
            
        print("Processing GDP Testing Data...")
        X_test_gdp_q = []
        for window in tqdm(X_test_gdp):
            features = qrc.get_features(window)
            X_test_gdp_q.append(features)
        
        # 4. Train & Predict
        model_gdp = Ridge(alpha=RIDGE_ALPHA)
        model_gdp.fit(X_train_gdp_q, y_train_gdp)
        predictions_gdp = model_gdp.predict(X_test_gdp_q)
        
        # 5. Evaluate
        rmse_gdp = np.sqrt(mean_squared_error(y_test_gdp, predictions_gdp))
        print(f"GDP RMSE: {rmse_gdp:.5f}")

        # Save Figure 4
        plt.figure(figsize=(12, 6))
        plt.plot(y_test_gdp, label='Actual GDP Volatility', color='blue', alpha=0.5)
        plt.plot(predictions_gdp, label='Quantum Prediction', color='red')
        plt.title(f'Quarterly GDP Growth Volatility (RMSE: {rmse_gdp:.5f})')
        plt.legend()
        plt.savefig("volatility_forecast_gdp.png")
        print("Graph saved as 'volatility_forecast_gdp.png'")
    else:
        print("Skipping GDP experiment due to data download error.")

if __name__ == "__main__":
    main()