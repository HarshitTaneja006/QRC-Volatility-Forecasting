import numpy as np

from src.quantum_reservoir import QuantumReservoir


def test_entangling_dynamics_change_reservoir_features():
    window = np.array([0.31, -0.47, 0.22])
    unentangled = QuantumReservoir(
        n_qubits=3,
        input_scale=1.0,
        layers=2,
        entangle=False,
        include_pair_observables=False,
    )
    entangled = QuantumReservoir(
        n_qubits=3,
        input_scale=1.0,
        layers=2,
        entangle=True,
        include_pair_observables=False,
    )

    assert not np.allclose(
        unentangled.get_features(window), entangled.get_features(window)
    )


def test_pair_observables_extend_feature_vector():
    reservoir = QuantumReservoir(
        n_qubits=4,
        input_scale=1.0,
        layers=1,
        include_pair_observables=True,
    )

    features = reservoir.get_features(np.array([0.1, 0.2, 0.3, 0.4]))

    assert features.shape == (8,)
    assert np.isfinite(features).all()


def test_global_depolarizing_noise_shrinks_traceless_observables():
    window = np.array([0.2, -0.1, 0.4, -0.3, 0.1])
    clean = QuantumReservoir(5, input_scale=0.8, layers=2)
    noisy = QuantumReservoir(5, input_scale=0.8, layers=2, depolarizing_probability=0.1)

    assert np.allclose(noisy.get_features(window), clean.get_features(window) * 0.9**2)
