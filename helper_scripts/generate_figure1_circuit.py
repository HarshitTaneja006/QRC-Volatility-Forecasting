"""

Figure 1: 5-Qubit QRC Circuit Diagram
Paper section: 3.4 (Amplitude Encoding of Classical Inputs)

Replicates the exact circuit built inside
src/quantum_reservoir.py :: QuantumReservoir.get_features()

Output: figures/figure1_quantum_circuit.png
"""

import os
import sys
import pathlib
import numpy as np

# ── allow imports from repo root ──────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit

# ── parameters (must match config.yaml exactly) ───────────────────────────────
N_QUBITS     = 5
SCALE_FACTOR = 80.0

# Representative normalised window — values chosen so Ry angles are
# spread across the Bloch sphere and visually distinct in the diagram.
# (actual forecasting uses real SPY returns; topology is identical)
DUMMY_WINDOW = np.array([0.10, 0.25, -0.15, 0.40, -0.30])

OUTPUT_DIR  = ROOT / "figures"
OUTPUT_FILE = OUTPUT_DIR / "figure1_quantum_circuit.png"

# ── build circuit (mirrors get_features logic exactly) ────────────────────────
qc = QuantumCircuit(N_QUBITS)

# Layer 1 — Amplitude encoding: Ry(S * x_i) on each qubit
for i in range(N_QUBITS):
    theta = round(SCALE_FACTOR * DUMMY_WINDOW[i], 3)
    qc.ry(theta, i)

qc.barrier()

# Layer 2 — Ring CZ entanglement: (q0,q1),(q1,q2),(q2,q3),(q3,q4),(q4,q0)
for i in range(N_QUBITS - 1):
    qc.cz(i, i + 1)
qc.cz(N_QUBITS - 1, 0)   # periodic boundary — closes the ring

qc.barrier()

# Layer 3 — Pauli-Z readout (shown explicitly so diagram matches paper fig)
qc.measure_all()

# ── draw ──────────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

fig = qc.draw(
    output="mpl",
    style="clifford",       # clean publication-grade colour scheme
    fold=-1,                # single unbroken row — no line wrapping
    plot_barriers=True,
    initial_state=True,     # show |0> labels on left
    reverse_bits=False,     # q_0 at top, q_4 at bottom
)

fig.suptitle(
    "Figure 1 — 5-Qubit QRC Architecture\n"
    "Layer 1: Rᵧ Encoding  |  Layer 2: CZ Ring Entanglement  |  Layer 3: Pauli-Z Readout",
    fontsize=10,
    y=1.04,
)

fig.savefig(OUTPUT_FILE, dpi=180, bbox_inches="tight")
plt.close(fig)

# ── verify ────────────────────────────────────────────────────────────────────
p = pathlib.Path(OUTPUT_FILE)
assert p.exists(),             f"ERROR: file not created at {p}"
assert p.stat().st_size > 5000, "ERROR: file suspiciously small — check renderer"

print(f"✅  Figure 1 saved  →  {p}")
print(f"    Size : {p.stat().st_size:,} bytes")
print(f"    Gates: {qc.count_ops()}")