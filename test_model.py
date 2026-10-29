import unittest

from simple_model import SimpleNeuralNetwork


class SimpleNeuralNetworkTest(unittest.TestCase):
    def test_learns_xor_outputs(self):
        samples = [
            ([0, 0], 0),
            ([0, 1], 1),
            ([1, 0], 1),
            ([1, 1], 0),
        ]
        model = SimpleNeuralNetwork(seed=7, hidden_size=6, learning_rate=0.08)
        model.train(samples, epochs=8000)

        predictions = {
            tuple(inputs): round(model.predict(inputs))
            for inputs, _ in samples
        }
        self.assertEqual(predictions[(0, 0)], 0)
        self.assertEqual(predictions[(0, 1)], 1)
        self.assertEqual(predictions[(1, 0)], 1)
        self.assertEqual(predictions[(1, 1)], 0)


if __name__ == "__main__":
    unittest.main()
