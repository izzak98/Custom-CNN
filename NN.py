from typing import Callable, Union
import cupy as np
from cupy import ndarray, float_
from tqdm import tqdm
from utils import convert_to_array
from lossFuncs import loss_derivative_mapping
from activationFuncs import act_derivative_mapping
from Optmizers import Adam


class NeuralNetwork:
    def __init__(self,
                 input_shape: Union[int, tuple[int, ...]],
                 lr=0.01,
                 loss=None
                 ) -> None:
        self.layers = []
        self.input_shape = input_shape
        self.lr = lr
        self.loss = loss
        self.optimizer = Adam(lr=lr)

    def get_weights(self) -> list[float_]:
        weights = []
        for layer in self.layers:
            try:
                weights.append(layer.weights)
                weights.append(layer.biases)
            except AttributeError:
                weights.append(None)
                weights.append(None)
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

    def add_layer(self, layer, **kwargs) -> None:
        if not self.layers:
            layer = layer(self.input_shape, **kwargs)
        else:
            layer = layer(self.layers[-1].output_size, **kwargs)
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
            output_size = layer.output_size
            if isinstance(output_size, tuple):
                a[i] = np.zeros((len(X), *output_size))
                z[i] = np.zeros((len(X), *output_size))
            else:
                a[i] = np.zeros((len(X),  layer.output_size))
                z[i] = np.zeros((len(X), layer.output_size))
        for l, _input in enumerate(X):
            for i, layer in enumerate(self.layers):
                if not hasattr(layer, 'weights'):
                    _input = layer(_input)
                    raw = _input
                else:
                    _input, raw = layer.forward(_input)
                a[i][l] = _input
                z[i][l] = raw
        return a, z

    def backwards_pass(self, X: np.ndarray, y: np.ndarray):
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        assert self.loss is not None, "Loss function not defined"

        # Forward pass to get activations and pre-activations
        a, z = self.forward_pass(X)
        gradients = {}

        # Initialize delta from the loss derivative w.r.t. the last layer's activation
        loss_prime = loss_derivative_mapping[self.loss]
        delta = loss_prime(a[len(self.layers) - 1], y)

        # Iterate backwards through the layers
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            gradients[f'W{i}'], gradients[f'b{i}'] = None, None  # For layers without weights

            # Backprop through layers with weights
            if hasattr(layer, 'weights'):
                if i == 0:
                    grad_weights, grad_bias, delta = layer.backward_pass(X, delta, False)
                else:
                    grad_weights, grad_bias, delta = layer.backward_pass(
                        a[i - 1], delta, True, z[i-1])
                grad_weights = grad_weights / X.shape[0]

                gradients[f'b{i}'] = grad_bias
                gradients[f'W{i}'] = grad_weights

            # Backprop through layers without weights
            elif hasattr(layer, 'backward'):
                delta = layer.backward(delta)
            # No else needed as we only update delta for layers that specifically require it

        return gradients, a[len(self.layers) - 1]

    def train(self,
              X: ndarray,
              y: ndarray,
              epochs: int,
              batch_size: int = 34,
              shuffle: bool = True) -> None:
        assert self.loss is not None, "Loss function not defined"
        X = convert_to_array(X)
        y = convert_to_array(y)

        if shuffle:
            indices = np.random.permutation(X.shape[0])
            X = X[indices]
            y = y[indices]

        progress = tqdm(range(epochs), desc='Training')
        for _ in progress:
            batch_loss = []
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                gradients, preds = self.backwards_pass(X_batch, y_batch)
                loss = self.loss(preds, y_batch)
                weights = self.get_weights()
                updated_weights = self.optimizer(weights, gradients)
                for i, layer in enumerate(self.layers):
                    if not hasattr(layer, 'weights'):
                        continue
                    layer.weights = updated_weights[f'W{i}']
                    layer.biases = updated_weights[f'b{i}']
                batch_loss.append(loss)
            progress.set_postfix(loss=round(float(np.mean(loss)), 6))


if __name__ == "__main__":
    from activationFuncs import relu, sigmoid
    from lossFuncs import mean_squared_error as mse
    from layers import DenseLayer, FlattenLayer, CnnLayer, MaxPoolingLayer
    nn = NeuralNetwork((5, 5, 3), lr=0.01, loss=mse)
    nn.add_layer(CnnLayer,
                 kernel_size=(3, 3),
                 filters=3,
                 stride=1,
                 padding=1,
                 activation=relu,
                 input_channels=3)
    nn.add_layer(MaxPoolingLayer, pool_size=2, stride=2)
    nn.add_layer(FlattenLayer)
    nn.add_layer(DenseLayer, output_size=10, activation=relu)
    nn.add_layer(DenseLayer, output_size=1, activation=sigmoid)
    X = np.random.rand(10, 5, 5, 3)
    y = np.random.rand(10)
    print(nn(X))
    nn.train(X, y, 1000)
