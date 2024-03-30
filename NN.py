from typing import Callable
import cupy as np
from cupy import ndarray, float_
from tqdm import tqdm
from utils import convert_to_array
from lossFuncs import loss_derivative_mapping
from activationFuncs import act_derivative_mapping


class DenseLayer:
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 activation: Callable[[ndarray], ndarray]
                 ) -> None:
        self.output_size = output_size
        self.weights = np.random.rand(output_size, input_size)
        self.biases = np.random.rand(output_size)
        self.activation = activation

    def __call__(self, inputs: ndarray) -> ndarray:
        z = np.dot(self.weights, inputs) + self.biases
        return self.activation(z)

    def forward(self, inputs: ndarray) -> ndarray:
        self.inputs = inputs
        self.z = np.dot(self.weights, inputs) + self.biases
        self.a = self.activation(self.z)
        return self.activation(self.z), self.z


class NeuralNetwork:
    def __init__(self, input_shape: int, lr=None, loss=None) -> None:
        self.layers = []
        self.input_shape = input_shape
        self.lr = lr
        self.loss = loss

    def get_weights(self) -> list[float_]:
        weights = []
        for layer in self.layers:
            weights.append(layer.weights)
            weights.append(layer.biases)
        return weights

    def load_weights(self, weights: list[float_]) -> None:
        for i, weight in enumerate(weights):
            weight = weight.T
            if i % 2 == 0:
                try:
                    weight = weight.reshape(self.layers[i // 2].weights.shape)
                except ValueError:
                    raise ValueError("Weight shape mismatch")
                weight = convert_to_array(weight)
                self.layers[i // 2].weights = weight
            elif i % 2 == 1:
                try:
                    weight = weight.reshape(self.layers[i // 2].biases.shape)
                except ValueError:
                    raise ValueError("Bias shape mismatch")
                weight = convert_to_array(weight)
                self.layers[i // 2].biases = weight

    def add_layer(self, shape: int, activation: Callable) -> None:
        if not self.layers:
            layer = DenseLayer(self.input_shape, shape, activation)
        else:
            layer = DenseLayer(
                self.layers[-1].output_size, shape, activation)
        self.layers.append(layer)

    def __call__(self, inputs: ndarray) -> ndarray:
        inputs = convert_to_array(inputs)
        outputs = []
        for _input in inputs:
            for layer in self.layers:
                _input = layer(_input)
            outputs.append(_input)
        outputs = np.array(outputs)
        return outputs

    def forward_pass(self, X: ndarray) -> ndarray:
        X = convert_to_array(X)
        a = {}
        z = {}
        for i, layer in enumerate(self.layers):
            a[i] = np.zeros((len(X),  layer.output_size))
            z[i] = np.zeros((len(X), layer.output_size))
        for l, _input in enumerate(X):
            for i, layer in enumerate(self.layers):
                _input, raw = layer.forward(_input)
                a[i][l] = _input
                z[i][l] = raw
        return a, z

    def backwards_pass(self, X: ndarray, y: ndarray):
        X = convert_to_array(X)
        y = convert_to_array(y).reshape(-1, 1)
        assert self.loss is not None, "Loss function not defined"
        a, z = self.forward_pass(X)

        gradients = {}

        loss_prime = loss_derivative_mapping[self.loss]
        delta = loss_prime(a[len(self.layers) - 1], y) * \
            act_derivative_mapping[self.layers[-1].activation](z[len(self.layers) - 1])

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            gradients[f'W{i}'], gradients[f'b{i}'] = [], []

            grad_bias = np.mean(delta, axis=0)
            grad_weights = np.dot(delta.T, a[i - 1] if i > 0 else X) / X.shape[0]

            gradients[f'b{i}'] = grad_bias
            gradients[f'W{i}'] = grad_weights

            if i > 0:
                delta = np.dot(delta, layer.weights) * \
                    act_derivative_mapping[self.layers[i - 1].activation](z[i - 1])

        return gradients

    def train(self, X: ndarray, y: ndarray, epochs: int) -> None:
        assert self.loss is not None, "Loss function not defined"
        X = convert_to_array(X)
        y = convert_to_array(y)

        progress = tqdm(range(epochs), desc='Training')
        for _ in progress:
            preds = self(X)
            loss = self.loss(y, preds)
            gradients = self.backwards_pass(X, y)
            for i, layer in enumerate(self.layers):
                layer.weights += self.lr * gradients[f'W{i}']
                layer.biases += self.lr * gradients[f'b{i}']
            progress.set_postfix(loss=round(float(loss), 6))


if __name__ == "__main__":
    from activationFuncs import relu, sigmoid
    from lossFuncs import mean_squared_error as mse
    nn = NeuralNetwork(3, lr=0.01, loss=mse)
    nn.add_layer(3, relu)
    nn.add_layer(3, relu)
    nn.add_layer(1, sigmoid)
    X = [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6]]
    y = [0.2, 0.3, 0.4, 0.5]
    nn.train(X, y, 10000)
    print(nn(X))
