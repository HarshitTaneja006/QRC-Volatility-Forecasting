import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

class QuantumReservoir:
    def __init__(self, n_qubits=5, scale_factor=80.0, shots=1024):
        """
        Initializes the Quantum Reservoir.
        
        Args:
            n_qubits (int): Number of qubits (window size).
            scale_factor (float): The "Volume Knob" for input scaling.
            shots (int): Number of simulation runs per data point.
        """
        self.n_qubits = n_qubits
        self.scale_factor = scale_factor
        self.shots = shots
        self.simulator = AerSimulator()

    def _build_circuit(self, input_data):
        """
        Constructs the Quantum Circuit for a given input window.
        """
        qc = QuantumCircuit(self.n_qubits)

        # LAYER 1: Encoding (Amplitude Encoding via Rotation)
        for i in range(self.n_qubits):
            # Scale input to activate non-linearity
            angle = input_data[i] * self.scale_factor
            qc.ry(angle, i)

        # LAYER 2: Reservoir (Entanglement via Ring Topology)
        # Mimics a 1D Transverse-Field Ising Spin Chain
        for i in range(self.n_qubits - 1):
            qc.cz(i, i + 1)
        # Close the loop (Periodic Boundary Condition)
        qc.cz(self.n_qubits - 1, 0)

        # LAYER 3: Measurement
        qc.measure_all()
        return qc

    def get_features(self, input_window):
        """
        Runs the circuit and returns the Expectation Values (Z-spin).
        """
        qc = self._build_circuit(input_window)
        job = self.simulator.run(qc, shots=self.shots)
        result = job.result()
        counts = result.get_counts()

        # Calculate Z-expectation values [-1, 1] for each qubit
        features = []
        for i in range(self.n_qubits):
            z_sum = 0
            for state, count in counts.items():
                # Qiskit uses Little-Endian ordering (reverse index)
                if state[self.n_qubits - 1 - i] == '0':
                    z_sum += count  # Spin Up (+1)
                else:
                    z_sum -= count  # Spin Down (-1)
            
            expectation = z_sum / self.shots
            features.append(expectation)
        
        return features