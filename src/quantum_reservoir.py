import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

class QuantumReservoir:
    def __init__(self, n_qubits=5, scale_factor=80.0, shots=1024, noise_model=None):
        self.n_qubits = n_qubits
        self.scale_factor = scale_factor
        self.shots = shots
        self.noise_model = noise_model

        self.simulator = AerSimulator(seed_simulator=42)

    def build_circuit(self, input_data):
        qc = QuantumCircuit(self.n_qubits)

        # --- Phase 1: Amplitude Encoding ---
        for i in range(self.n_qubits):
            angle = input_data[i] * self.scale_factor
            qc.ry(angle, i)

        # --- Phase 2: Ring Entanglement ---
        for i in range(self.n_qubits - 1):
            qc.cz(i, i + 1)

        qc.cz(self.n_qubits - 1, 0)

        qc.measure_all()
        return qc

    def get_features(self, input_data):
        qc = self.build_circuit(input_data)

        job = self.simulator.run(
            qc,
            shots=self.shots,
            noise_model=self.noise_model
        )

        counts = job.result().get_counts()

        features = []
        for i in range(self.n_qubits):
            z_sum = 0
            for state, count in counts.items():
                if state[self.n_qubits - 1 - i] == '0':
                    z_sum += count
                else:
                    z_sum -= count
            features.append(z_sum / self.shots)

        return features