# Real-Time Macroeconomic Volatility Forecasting using Quantum Reservoir Computing

![License](https://img.shields.io/badge/license-MIT-blue.svg) ![Python](https://img.shields.io/badge/python-3.8%2B-blue) ![Qiskit](https://img.shields.io/badge/qiskit-1.0%2B-purple)

Official implementation of the research project: **"Real-Time Macroeconomic Volatility Forecasting: A Comparative Study of Quantum Reservoir Computing vs. GARCH Models"**.

## 📌 Project Overview

This project investigates the efficacy of **Quantum Reservoir Computing (QRC)** for predicting the daily volatility of the S&P 500 index. By encoding financial time-series data into a 5-qubit Transverse-Field Ising Model, we demonstrate that quantum non-linearity can capture market dynamics better than traditional linear baselines.

**Key Findings:**
- **Baseline (GARCH):** Robust but slow to react to sudden shocks
- **Quantum Model (Tuned):** Achieved an RMSE of **0.00508**, outperforming un-tuned baselines by ~21%

## 🎯 Key Features

- **Quantum Circuit Simulation** using Qiskit's Aer backend
- **Transverse-Field Ising Model** implementation with ring topology
- **Amplitude Encoding** via rotation gates for time-series data
- **Ridge Regression** readout layer for volatility prediction
- **Real-time market data** integration via Yahoo Finance API

## 📂 Repository Structure

```text
QRC-Volatility-Forecasting/
├── src/
│   ├── quantum_reservoir.py  # Quantum circuit logic and feature extraction
│   └── data_processor.py     # Data downloading and preprocessing
├── main.py                   # Main experiment script
├── requirements.txt          # Project dependencies
├── README.md                 # Project documentation
└── LICENSE                   # MIT License
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/harshittaneja006/QRC-Volatility-Forecasting.git
cd QRC-Volatility-Forecasting
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Experiment

To download data, train the quantum model, and generate the forecast graph:

```bash
python main.py
```

### Expected Output

The script will:
1. Download S&P 500 data from Yahoo Finance
2. Process and create sliding windows for time-series analysis
3. Run quantum circuit simulations for feature extraction
4. Train the Ridge regression readout layer
5. Print RMSE evaluation metrics
6. Generate and save a comparison plot (`volatility_forecast.png`)

## 🔬 Methodology

### 1. Data Processing
- Downloads historical S&P 500 (SPY) data
- Calculates log returns: `ln(P_t / P_{t-1})`
- Creates sliding windows of size 5 for temporal analysis

### 2. Quantum Reservoir Architecture

**Layer 1: Encoding**
- Amplitude encoding via RY rotation gates
- Input scaling factor: 80.0 (optimized through hyperparameter tuning)

**Layer 2: Reservoir Dynamics**
- 5-qubit Transverse-Field Ising Model
- Ring topology with CZ entanglement gates
- Periodic boundary conditions

**Layer 3: Measurement**
- Z-basis measurement on all qubits
- Expectation values used as features

### 3. Readout Layer
- Ridge regression (α = 0.01)
- Maps quantum features to volatility predictions

## 🛠️ Technologies

| Technology | Purpose |
|------------|---------|
| **Qiskit** | Quantum circuit construction and simulation |
| **Qiskit Aer** | High-performance quantum simulator backend |
| **Scikit-Learn** | Ridge regression and evaluation metrics |
| **YFinance** | Market data acquisition |
| **NumPy** | Numerical computations |
| **Pandas** | Data manipulation |
| **Matplotlib** | Visualization |

## 📊 Results

The quantum reservoir model demonstrates superior performance in capturing market volatility:

- **Training Set:** 80% of historical data
- **Testing Set:** 20% of historical data
- **Final RMSE:** 0.00508
- **Improvement:** ~21% over un-tuned baseline

## ⚙️ Configuration

Key hyperparameters can be adjusted in `main.py`:

```python
TICKER = 'SPY'           # Stock ticker symbol
WINDOW_SIZE = 5          # Number of qubits / lookback period
SCALE_FACTOR = 80.0      # Input scaling for quantum encoding
RIDGE_ALPHA = 0.01       # Ridge regression regularization
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{qrc_volatility_2024,
  title={Real-Time Macroeconomic Volatility Forecasting: A Comparative Study of Quantum Reservoir Computing vs. GARCH Models},
  author={Your Name},
  year={2024}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- [Qiskit Documentation](https://qiskit.org/documentation/)
- [Quantum Reservoir Computing Paper](https://arxiv.org/abs/your-paper-id)
- [Report Issues](https://github.com/YOUR_USERNAME/QRC-Volatility-Forecasting/issues)

## 📧 Contact

For questions or collaboration opportunities, please open an issue or contact [harshit.taneja2025@vitstudent.ac.in]

---
