from qiskit_aer.noise import NoiseModel, depolarizing_error

def create_noise_model(p=0.01):
    noise_model = NoiseModel()

    error = depolarizing_error(p, 1)
    noise_model.add_all_qubit_quantum_error(error, ['ry'])

    return noise_model
