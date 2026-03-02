import numpy as np

class ClassicalReservoir:
    def __init__(self, n_features=5, reservoir_size=50):
        np.random.seed(42)

        self.reservoir_size = reservoir_size
        self.W = np.random.randn(reservoir_size, reservoir_size) * 0.1
        self.Win = np.random.randn(reservoir_size, n_features)

    def get_features(self, input_data):
        state = np.zeros(self.reservoir_size)

        for x in input_data:
            state = np.tanh(self.W @ state + self.Win @ input_data)

        return state
