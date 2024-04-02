"""This module contains classes for different types of layers in a neural network."""
from typing import Callable, Optional, Any
import cupy as np
from cupy import ndarray
from utils import convert_to_array
from activationFuncs import act_derivative_mapping


class DenseLayer:
    """A class to represent a dense layer in a neural network."""

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 activation: Callable[[ndarray], ndarray]
                 ) -> None:
        """
        Initialize a Layer object.

        Args:
            input_size (int): The size of the input to the layer.
            output_size (int): The size of the output from the layer.
            activation (Callable[[ndarray], ndarray]): The activation function to be used.

        Returns:
            None
        """
        self.output_size = output_size
        self.weights = np.random.rand(output_size, input_size)
        self.biases = np.random.rand(output_size)
        self.activation = activation
        self.act_derivative = act_derivative_mapping[self.activation]  # type: ignore
        self.soxtmax = self.activation.__name__ == "softmax"

    def __call__(self, inputs: ndarray) -> ndarray:
        """
        Applies the layer's weights and biases to the input and returns the result
        after applying the activation function.

        Args:
            inputs (ndarray): The input array to the layer.

        Returns:
            ndarray: The output array after applying the layer's weights, 
            biases, and activation function.
        """
        z = np.dot(self.weights, inputs) + self.biases
        return self.activation(z)

    def forward(self, inputs: ndarray) -> ndarray:
        """
        Performs the forward pass of the layer.

        Args:
            inputs (ndarray): The input data.

        Returns:
            ndarray: The output of the layer after applying the activation function.
            ndarray: The pre-activation values before applying the activation function.
        """
        z = np.dot(self.weights, inputs) + self.biases
        return self.activation(z), z

    def backward_pass(self,
                      prev_output,
                      delta,
                      calc_next_delta=True,
                      prev_raw_output=None) -> tuple[ndarray, ndarray, Optional[ndarray]]:
        """
        Performs the backward pass of the layer.

        Args:
            prev_output (numpy.ndarray): The output from the previous layer.
            delta (numpy.ndarray): The error gradient with respect to the layer's output.
            calc_next_delta (bool, optional): Whether to calculate the error gradient with 
            respect to the previous layer's output. Defaults to True.
            prev_raw_output (numpy.ndarray, optional): The raw output from the previous layer. 
            Defaults to None.

        Returns:
            tuple: A tuple containing the gradient of the weights, the gradient of the biases,
            and the error gradient with respect to the previous 
            layer's output (if calc_next_delta is True).
        """
        grad_weights = np.dot(prev_output.T, delta)
        grad_biases = np.sum(delta, axis=0, keepdims=True)
        if calc_next_delta:
            next_delta = np.dot(delta, self.weights)
            if not self.soxtmax:
                next_delta *= self.act_derivative(prev_raw_output)
        else:
            next_delta = None
        return grad_weights.T, grad_biases, next_delta


class FlattenLayer:
    """A class to represent a flatten layer in a neural network."""

    def __init__(self,
                 input_size: tuple[int, ...]) -> None:
        """
        Initialize a FlattenLayer object.

        Args:
            input_size (tuple[int, ...]): The size of the input to the layer.

        Returns:
            None"""
        self.output_size = 1
        for dim in input_size:
            self.output_size *= dim
        self.shape_cache = input_size

    def __call__(self, inputs: ndarray) -> ndarray:
        """
        Applies the layer to the given inputs.

        Args:
            inputs (ndarray): The input data.

        Returns:
            ndarray: The flattened input data.
        """
        inputs = convert_to_array(inputs)
        return inputs.flatten()

    def backward(self, d_out: ndarray) -> ndarray:
        """
        Performs the backward pass of the layer.

        Args:
            d_out (ndarray): The gradient of the loss with respect to the output of the layer.

        Returns:
            ndarray: The gradient of the loss with respect to the input of the layer.
        """
        new_d_out = []
        for d_out_val in d_out:
            new_d_out.append(d_out_val.reshape(self.shape_cache))
        return np.array(new_d_out)


class CnnLayer:
    """A class to represent a convolutional layer in a neural network."""

    def __init__(self,
                 input_size: tuple[int, ...],
                 kernel_size: tuple[int, int],
                 filters: int,
                 stride: int,
                 activation: Callable[[ndarray], ndarray],
                 padding: int = 0,
                 input_channels: int = 1  # Assuming 1 input channel by default
                 ) -> None:
        """
        Initializes a custom CNN layer.

        Args:
            input_size (tuple[int, ...]): The size of the input tensor.
            kernel_size (tuple[int, int]): The size of the kernel/filter.
            filters (int): The number of filters in the layer.
            stride (int): The stride value for the convolution operation.
            activation (Callable[[ndarray], ndarray]): The activation function to
            be applied to the output.
            padding (int, optional): The amount of padding to be applied to the input.
            Defaults to 0.
            input_channels (int, optional): The number of input channels.
            Defaults to 1.

        Returns:
            None
        """
        assert kernel_size < input_size, "Kernel size must be smaller than input size"
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.filters = filters
        self.activation = activation
        self.stride = stride
        self.padding = padding
        self.input_channels = input_channels
        self.output_size = (1 + (input_size[0] - kernel_size[0] + 2 * padding) // stride,
                            1 + (input_size[1] - kernel_size[1] + 2 * padding) // stride,
                            filters)
        self.weights = np.random.rand(filters, input_channels, kernel_size[0], kernel_size[1])/10
        self.biases = np.random.rand(filters)/10

    def pad(self, inputs: ndarray) -> ndarray:
        """
        Pads the input array with zeros.

        Args:
            inputs (ndarray): The input array to be padded.

        Returns:
            ndarray: The padded array.
        """
        padded_inputs = np.pad(inputs, ((self.padding, self.padding),
                                        (self.padding, self.padding), (0, 0)), 'constant')
        return padded_inputs

    def convolve(self, inputs: ndarray) -> ndarray:
        """
        Applies convolution operation on the given inputs.

        Args:
            inputs (ndarray): The input array of shape (height, width, channels).

        Returns:
            ndarray: The output array after convolution, of shape (output_height,
            output_width, filters).
        """
        padded_inputs = self.pad(inputs) if self.padding > 0 else inputs
        output_height, output_width = self.output_size[0], self.output_size[1]
        output = np.zeros((output_height, output_width, self.filters))
        for filter_index in range(self.filters):
            for i in range(output_height):
                for j in range(output_width):
                    row_start, col_start = i * self.stride, j * self.stride
                    snippet = padded_inputs[row_start:row_start + self.kernel_size[0],
                                            col_start:col_start+self.kernel_size[1],
                                            :]
                    output[i, j, filter_index] = np.sum(
                        snippet * self.weights[filter_index]) + self.biases[filter_index]
        return output

    def __call__(self, inputs: ndarray) -> ndarray:
        """
        Applies the convolutional layer to the given inputs.

        Args:
            inputs (ndarray): The input array to apply the convolution on.

        Returns:
            ndarray: The output array after applying the convolution.
        """
        if inputs.ndim < 3:
            inputs = np.expand_dims(inputs, axis=-1)
        output = self.convolve(inputs)
        if self.activation:
            output = self.activation(output)
        return output

    def forward(self, inputs: ndarray) -> ndarray:
        """
        Performs the forward pass of the layer.

        Args:
            inputs (ndarray): The input data.

        Returns:
            ndarray: The output of the layer after applying the convolution and
            activation functions.
        """
        if inputs.ndim < 3:
            inputs = np.expand_dims(inputs, axis=-1)
        z = self.convolve(inputs)
        return self.activation(z), z

    def backward_pass(self,
                      prev_output: ndarray,
                      delta: ndarray,
                      calc_next_delta: bool = True,
                      prev_raw_output: Any = None) -> tuple[ndarray, ndarray, Optional[ndarray]]:
        """
        Performs the backward pass of the layer.

        Args:
            prev_output (ndarray): The output of the previous layer.
            delta (ndarray): The delta (gradient) of the loss function with respect to
            the output of this layer.
            calc_next_delta (bool, optional): Whether to calculate the delta of the loss function 
            with respect to the input of the previous layer. Defaults to True.
            prev_raw_output (Any, optional): The raw output of the previous layer. 
            Defaults to None.

        Returns:
            tuple[ndarray, ndarray, Optional[ndarray]]: A tuple containing the gradients
            of the weights, biases, and the next delta (if calc_next_delta is True).
        """
        _, _, _, channels = prev_output.shape
        prev_output_padded = np.array([self.pad(p_o) for p_o in prev_output])
        grad_weights = np.zeros_like(self.weights)
        grad_biases = np.zeros_like(self.biases)

        # Compute the gradients of the weights and biases
        for f in range(self.filters):
            for c in range(channels):
                for i in range(self.kernel_size[0]):
                    for j in range(self.kernel_size[1]):
                        grad_weights[f, c, i, j] = np.sum(
                            delta[:, :, :, f] * prev_output_padded[:,
                                                                   i:i+delta.shape[1],
                                                                   j:j+delta.shape[2],
                                                                   c]
                        )
            grad_biases[f] = np.sum(delta[:, :, :, f])

        # Compute the gradient of the loss function with respect to the input of this layer
        if calc_next_delta:
            next_delta = np.zeros_like(prev_output)
            for f in range(self.filters):
                for c in range(channels):
                    padded_delta = np.pad(delta[:, :, :, f],
                                          ((self.kernel_size[0] // 2, self.kernel_size[0] // 2),
                                           (self.kernel_size[1] // 2, self.kernel_size[1] // 2)),
                                          'constant')
                    for i in range(prev_output.shape[1]):
                        for j in range(prev_output.shape[2]):
                            next_delta[:, i, j, c] += np.sum(
                                padded_delta[:, i:i+self.kernel_size[0], j:j+self.kernel_size[1]] *
                                self.weights[f, :, :, c]
                            )
        else:
            next_delta = None

        return grad_weights, grad_biases, next_delta


class MaxPoolingLayer:
    """A class to represent a max pooling layer in a neural network."""

    def __init__(self, input_size: tuple[int, ...], pool_size: int, stride: int) -> None:
        """
        Initializes a PoolingLayer object.

        Args:
            input_size (tuple[int, ...]): The size of the input tensor.
            pool_size (int): The size of the pooling window.
            stride (int): The stride value for the pooling operation.

        Attributes:
            pool_size (int): The size of the pooling window.
            stride (int): The stride value for the pooling operation.
            output_size (tuple[int, ...]): The size of the output tensor after pooling.
            max_indices (None): Placeholder for storing the indices
            of the maximum values during pooling.
        """
        self.pool_size = pool_size
        self.stride = stride
        self.output_size = (1 + (input_size[0] - pool_size) // stride,
                            1 + (input_size[1] - pool_size) // stride,
                            input_size[2])
        self.max_indices = None

    def __call__(self, inputs: ndarray) -> ndarray:
        """
        Applies max pooling operation to the input tensor.

        Args:
            inputs (ndarray): The input tensor of shape (n_h, n_w, n_c).

        Returns:
            ndarray: The output tensor after applying max pooling, of shape (h_out, w_out, n_c).
        """
        (n_h, n_w, n_c) = inputs.shape
        h_out = 1 + (n_h - self.pool_size) // self.stride
        w_out = 1 + (n_w - self.pool_size) // self.stride
        output = np.zeros((h_out, w_out, n_c))

        # Initialize max_indices to track the position of max values
        self.max_indices = np.zeros((h_out, w_out, n_c, 2), dtype=int)

        for h in range(h_out):
            for w in range(w_out):
                for c in range(n_c):
                    h_start = h * self.stride
                    h_end = h_start + self.pool_size
                    w_start = w * self.stride
                    w_end = w_start + self.pool_size

                    window = inputs[h_start:h_end, w_start:w_end, c]
                    max_val = np.max(window)
                    output[h, w, c] = max_val
                    # Find the index of the max value within the window
                    max_index = np.unravel_index(np.argmax(window, axis=None), window.shape)
                    self.max_indices[h, w, c] = np.array(
                        [h_start + max_index[0], w_start + max_index[1]])

        return output

    def backward(self, d_out: ndarray) -> ndarray:
        """
        Performs the backward pass of the pooling layer.

        Args:
            d_out (ndarray): The gradient of the loss with respect to the output
            of the pooling layer.

        Returns:
            ndarray: The gradient of the loss with respect to the input of the pooling layer.
        """
        assert self.max_indices is not None, "Max indices not initialized"
        d_inputs = []
        for d_out_val in d_out:
            (n_h, n_w, n_c) = d_out_val.shape
            d_input = np.zeros((n_h * self.stride + self.pool_size - 1,
                                n_w * self.stride + self.pool_size - 1,
                                n_c))
            for h in range(n_h):
                for w in range(n_w):
                    for c in range(n_c):
                        (h_max, w_max) = self.max_indices[h, w, c]
                        # Route the gradient to the max location
                        d_input[h_max, w_max, c] += d_out_val[h, w, c]
            d_inputs.append(d_input)
        return np.array(d_inputs)
