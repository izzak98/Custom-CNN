"""A module containing the NeuralNetwork class."""
from typing import Union
import cupy as np
from cupy import ndarray, float_
from tqdm import tqdm
from utils import convert_to_array
from lossFuncs import loss_derivative_mapping
from Optmizers import Adam


class NeuralNetwork:
    """A class to represent a neural network model."""

    def __init__(self,
                 input_shape: Union[int, tuple[int, ...]],
                 lr=0.01,
                 loss=None
                 ) -> None:
        """
        Initializes a NeuralNetwork object.

        Args:
            input_shape (Union[int, tuple[int, ...]]): The shape of the input data.
            lr (float, optional): The learning rate for the optimizer. Defaults to 0.01.
            loss (callable, optional): The loss function to be used. Defaults to None.
        """
        self.layers = []
        self.input_shape = input_shape
        self.lr = lr
        self.loss = loss
        self.optimizer = Adam(lr=lr)
        self.weighted_layers = []

    def get_weights(self) -> list[ndarray]:
        """
        Returns the weights of all the layers in the neural network.

        Returns:
            list[ndarray]: The weights of all the layers.
        """
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
        """
        Loads the weights into the neural network.

        Args:
            weights (list[float_]): The weights to be loaded.
        """
        for i, weight in enumerate(weights):
            weight = weight.T
            if i % 2 == 0:
                try:
                    weight = weight.reshape(self.weighted_layers[i // 2].weights.shape)
                except ValueError:
                    raise ValueError("Weight shape mismatch")
                weight = convert_to_array(weight)
                self.weighted_layers[i // 2].weights = weight
            elif i % 2 == 1:
                try:
                    weight = weight.reshape(self.weighted_layers[i // 2].biases.shape)
                except ValueError:
                    raise ValueError("Bias shape mismatch")
                weight = convert_to_array(weight)
                self.weighted_layers[i // 2].biases = weight

    def add_layer(self, layer, **kwargs) -> None:
        """
        Adds a layer to the neural network.

        Args:
            layer: The layer to be added.
            **kwargs: Additional keyword arguments for the layer.
        """
        if not self.layers:
            layer = layer(self.input_shape, **kwargs)
        else:
            layer = layer(self.layers[-1].output_size, **kwargs)
        self.layers.append(layer)
        if hasattr(layer, 'weights'):
            self.weighted_layers.append(layer)

    def __call__(self, inputs: ndarray) -> ndarray:
        """
        Performs a forward pass through the neural network.

        Args:
            inputs (ndarray): The input data.

        Returns:
            ndarray: The output of the neural network.
        """
        inputs = convert_to_array(inputs)
        outputs = []
        for _input in inputs:
            for layer in self.layers:
                _input = layer(_input)
            outputs.append(_input)
        outputs = np.array(outputs)
        return outputs

    def forward_pass(self, X: ndarray) -> tuple[dict[int, ndarray], dict[int, ndarray]]:
        """
        Performs a forward pass through the neural network.

        Args:
            X (ndarray): The input data.

        Returns:
            tuple[dict[int, ndarray], dict[int, ndarray]]: A tuple containing two dictionaries.
            The first dictionary `a` contains the activation values for each layer.
            The second dictionary `z` contains the raw output values for each layer.
        """
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

    def backwards_pass(self, X: ndarray, y: ndarray) -> tuple[dict[str, ndarray], ndarray]:
        """
        Perform the backward pass of the neural network.

        Args:
            X (ndarray): The input data.
            y (ndarray): The target labels.

        Returns:
            tuple[Dict[str, ndarray], ndarray]: A tuple containing the gradients
            of the weights and biases, and the activations of the last layer.

        Raises:
            AssertionError: If the loss function is not defined.
        """
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        assert self.loss is not None, "Loss function not defined"

        # Forward pass to get activations and pre-activations
        a, z = self.forward_pass(X)
        gradients = {}

        # Initialize delta from the loss derivative w.r.t. the last layer's activation
        loss_prime = loss_derivative_mapping[self.loss]
        delta = loss_prime(y, a[len(self.layers) - 1])

        # Iterate backwards through the layers
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            gradients[f'W{i}'], gradients[f'b{i}'] = None, None  # For layers without weights

            # Backprop through layers with weights
            if hasattr(layer, 'weights'):
                if i == 0:
                    grad_weights, grad_bias, delta = layer.backward_pass(X, delta, False)
                else:
                    if layer.soxtmax:
                        grad_weights, grad_bias, delta = layer.backward_pass(
                            a[i - 1], delta, True, y)
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
        """
        Trains the neural network model using the given input data.

        Args:
            X (ndarray): The input data.
            y (ndarray): The target labels.
            epochs (int): The number of training epochs.
            batch_size (int, optional): The batch size for training. Defaults to 34.
            shuffle (bool, optional): Whether to shuffle the data before each epoch. 
            Defaults to True.
        """
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
                loss = self.loss(y_batch, preds)
                loss = self.loss(y_batch, preds)
                weights = self.get_weights()
                updated_weights = self.optimizer(weights, gradients)
                for i, layer in enumerate(self.layers):
                    if not hasattr(layer, 'weights'):
                        continue
                    layer.weights = updated_weights[f'W{i}']
                    layer.biases = updated_weights[f'b{i}']
                batch_loss.append(loss)
            progress.set_postfix(loss=round(float(np.mean(loss)), 6))
