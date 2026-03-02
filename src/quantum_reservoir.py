import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

class QuantumReservoir:

    def __init__(self, n_qubits, scale_factor):
        self.n_qubits = n_qubits
        self.scale_factor = scale_factor

    def get_features(self, window):

        qc = QuantumCircuit(self.n_qubits)

        # Encoding
        for i in range(self.n_qubits):
            theta = self.scale_factor * window[i]
            qc.ry(theta, i)

        # Ring entanglement
        for i in range(self.n_qubits - 1):
            qc.cz(i, i+1)
        qc.cz(self.n_qubits - 1, 0)

        # Statevector expectation
        state = Statevector.from_instruction(qc)

        features = []
        for i in range(self.n_qubits):
            z_exp = state.expectation_value(
                [[1,0],[0,-1]], [i]
            )
            features.append(np.real(z_exp))

        return features