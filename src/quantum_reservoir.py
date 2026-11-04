"""Fixed quantum reservoir feature maps for rolling forecasting experiments."""

import numpy as np
from qiskit import QuantumCircuit


class QuantumReservoir:
    """Angle-encoded quantum feature map with non-commuting entangling dynamics."""

    def __init__(
        self,
        n_qubits: int,
        input_scale: float | None = None,
        *,
        scale_factor: float | None = None,
        layers: int = 2,
        mixing_angle: float = 0.7,
        entangle: bool = True,
        include_pair_observables: bool = True,
        depolarizing_probability: float = 0.0,
    ):
        if n_qubits < 2:
            raise ValueError("QuantumReservoir requires at least two qubits")
        if layers < 1:
            raise ValueError("layers must be at least one")
        if input_scale is None:
            input_scale = scale_factor
        if input_scale is None:
            raise ValueError("input_scale is required")
        if not 0.0 <= depolarizing_probability < 1.0:
            raise ValueError("depolarizing_probability must be in [0, 1)")

        self.n_qubits = n_qubits
        self.input_scale = float(input_scale)
        self.layers = layers
        self.mixing_angle = float(mixing_angle)
        self.entangle = entangle
        self.include_pair_observables = include_pair_observables
        self.depolarizing_probability = float(depolarizing_probability)
        self._dimension = 2**n_qubits
        basis = np.arange(self._dimension)
        self._z_signs = np.vstack(
            [1.0 - 2.0 * ((basis >> qubit) & 1) for qubit in range(n_qubits)]
        )

    def _build_circuit(self, window: np.ndarray) -> QuantumCircuit:
        if len(window) != self.n_qubits:
            raise ValueError(
                f"window length must equal n_qubits ({self.n_qubits}), got {len(window)}"
            )

        circuit = QuantumCircuit(self.n_qubits)
        for _ in range(self.layers):
            for qubit, value in enumerate(window):
                circuit.ry(self.input_scale * float(value), qubit)
            for qubit in range(self.n_qubits):
                circuit.rx(self.mixing_angle, qubit)
            if self.entangle:
                for qubit in range(self.n_qubits - 1):
                    circuit.cx(qubit, qubit + 1)
                circuit.cx(self.n_qubits - 1, 0)
        return circuit

    def transform(self, windows: np.ndarray) -> np.ndarray:
        """Return reservoir features for a two-dimensional batch of input windows."""
        windows = np.asarray(windows, dtype=float)
        if windows.ndim != 2 or windows.shape[1] != self.n_qubits:
            raise ValueError("windows must have shape (n_samples, n_qubits)")
        n_samples = len(windows)
        state = np.zeros((n_samples, self._dimension), dtype=complex)
        state[:, 0] = 1.0
        indices = np.arange(self._dimension)
        for _ in range(self.layers):
            for qubit in range(self.n_qubits):
                mask = 1 << qubit
                lower = indices[(indices & mask) == 0]
                upper = lower + mask
                angle = self.input_scale * windows[:, qubit] / 2.0
                cosine, sine = np.cos(angle)[:, None], np.sin(angle)[:, None]
                lower_state, upper_state = state[:, lower].copy(), state[:, upper].copy()
                state[:, lower] = cosine * lower_state - sine * upper_state
                state[:, upper] = sine * lower_state + cosine * upper_state
            cosine, sine = np.cos(self.mixing_angle / 2.0), np.sin(self.mixing_angle / 2.0)
            for qubit in range(self.n_qubits):
                mask = 1 << qubit
                lower = indices[(indices & mask) == 0]
                upper = lower + mask
                lower_state, upper_state = state[:, lower].copy(), state[:, upper].copy()
                state[:, lower] = cosine * lower_state - 1j * sine * upper_state
                state[:, upper] = -1j * sine * lower_state + cosine * upper_state
            if self.entangle:
                for control, target in [
                    *[(qubit, qubit + 1) for qubit in range(self.n_qubits - 1)],
                    (self.n_qubits - 1, 0),
                ]:
                    control_mask, target_mask = 1 << control, 1 << target
                    lower_target = indices[
                        ((indices & control_mask) != 0) & ((indices & target_mask) == 0)
                    ]
                    upper_target = lower_target + target_mask
                    temporary = state[:, lower_target].copy()
                    state[:, lower_target] = state[:, upper_target]
                    state[:, upper_target] = temporary
        probabilities = np.abs(state) ** 2
        observables = [self._z_signs[qubit] for qubit in range(self.n_qubits)]
        if self.include_pair_observables:
            observables.extend(
                self._z_signs[qubit] * self._z_signs[(qubit + 1) % self.n_qubits]
                for qubit in range(self.n_qubits)
            )
        return probabilities @ np.vstack(observables).T * (1.0 - self.depolarizing_probability) ** self.layers

    def get_features(self, window: np.ndarray) -> np.ndarray:
        """Return local-Z and optional nearest-neighbour ZZ expectations."""
        window = np.asarray(window, dtype=float)
        if len(window) != self.n_qubits:
            raise ValueError(
                f"window length must equal n_qubits ({self.n_qubits}), got {len(window)}"
            )
        return self.transform(window.reshape(1, -1))[0]
