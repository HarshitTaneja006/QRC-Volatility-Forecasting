import yaml
import numpy as np
import random

from src.data_loader import load_spy
from src.preprocessing import normalize_series
from src.quantum_reservoir import QuantumReservoir
from src.experiment_runner import run_qrc
from src.garch_model import run_garch
from src.metrics import rmse, diebold_mariano

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Fix seeds
np.random.seed(config["random_seed"])
random.seed(config["random_seed"])

# Load data
spy = load_spy(
    config["data"]["financial"]["start"],
    config["data"]["financial"]["end"]
)

split = int(len(spy) * config["train_split"])

train, test = normalize_series(spy, split)

# Quantum reservoir
reservoir = QuantumReservoir(
    n_qubits=config["window_size"],
    scale_factor=config["scale_factor"]
)

# Run QRC
y_test, qrc_preds = run_qrc(train, test, config, reservoir)

# Run GARCH
garch_preds = run_garch(train, test)

# Metrics
print("QRC RMSE:", rmse(y_test, qrc_preds))
print("GARCH RMSE:", rmse(y_test, garch_preds))

dm, p = diebold_mariano(y_test, garch_preds, qrc_preds)
print("DM statistic:", dm)
print("p-value:", p)