import math
import random


class SimpleNeuralNetwork:
    """Uma rede neural pequena, treinada pelo gradiente descendente."""

    def __init__(self, seed=42, hidden_size=6, learning_rate=0.08):
        random.seed(seed)
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.weights_input_hidden = [
            [random.uniform(-0.5, 0.5) for _ in range(hidden_size)]
            for _ in range(2)
        ]
        self.bias_hidden = [random.uniform(-0.5, 0.5) for _ in range(hidden_size)]
        self.weights_hidden_output = [random.uniform(-0.5, 0.5) for _ in range(hidden_size)]
        self.bias_output = random.uniform(-0.5, 0.5)

    @staticmethod
    def sigmoid(value):
        if value >= 0:
            return 1.0 / (1.0 + math.exp(-value))
        exp_value = math.exp(value)
        return exp_value / (1.0 + exp_value)

    @staticmethod
    def tanh(value):
        return math.tanh(value)

    def forward(self, inputs):
        hidden = [
            self.tanh(
                inputs[0] * self.weights_input_hidden[0][i]
                + inputs[1] * self.weights_input_hidden[1][i]
                + self.bias_hidden[i]
            )
            for i in range(self.hidden_size)
        ]
        output_value = sum(
            hidden[i] * self.weights_hidden_output[i]
            for i in range(self.hidden_size)
        ) + self.bias_output
        return hidden, self.sigmoid(output_value)

    def train(self, samples, epochs=6000):
        losses = []
        for epoch in range(epochs):
            total_error = 0.0

            for inputs, target in samples:
                hidden, output = self.forward(inputs)

                output_error = output - target
                output_delta = output_error * output * (1.0 - output)
                total_error += output_error * output_error

                hidden_errors = [
                    output_delta * self.weights_hidden_output[i]
                    for i in range(self.hidden_size)
                ]

                for i in range(self.hidden_size):
                    for input_index in range(2):
                        self.weights_input_hidden[input_index][i] -= (
                            self.learning_rate
                            * hidden_errors[i]
                            * (1.0 - hidden[i] * hidden[i])
                            * inputs[input_index]
                        )
                    self.bias_hidden[i] -= (
                        self.learning_rate
                        * hidden_errors[i]
                        * (1.0 - hidden[i] * hidden[i])
                    )

                for i in range(self.hidden_size):
                    self.weights_hidden_output[i] -= (
                        self.learning_rate * output_delta * hidden[i]
                    )
                self.bias_output -= self.learning_rate * output_delta

            if epoch % 500 == 0 or epoch == epochs - 1:
                losses.append(round(total_error / len(samples), 6))

        return losses

    def predict(self, inputs):
        _, probability = self.forward(inputs)
        return probability

    def graph(self):
        return {
            "inputNodes": [
                {"id": "x1", "label": "X", "layer": 0},
                {"id": "x2", "label": "Y", "layer": 0},
            ],
            "hiddenNodes": [
                {"id": f"h{i + 1}", "label": f"H{i + 1}", "layer": 1}
                for i in range(self.hidden_size)
            ],
            "outputNodes": [
                {"id": "output", "label": "Classe", "layer": 2}
            ],
            "inputHiddenEdges": [
                {
                    "source": f"x{input_index + 1}",
                    "target": f"h{hidden_index + 1}",
                    "weight": round(
                        self.weights_input_hidden[input_index][hidden_index], 5
                    ),
                }
                for input_index in range(2)
                for hidden_index in range(self.hidden_size)
            ],
            "hiddenOutputEdges": [
                {
                    "source": f"h{hidden_index + 1}",
                    "target": "output",
                    "weight": round(self.weights_hidden_output[hidden_index], 5),
                }
                for hidden_index in range(self.hidden_size)
            ],
        }
