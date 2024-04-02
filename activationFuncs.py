"""Activation functions and their derivatives."""
import cupy as np
from cupy import ndarray


def relu(x: ndarray) -> ndarray:
    """Compute the ReLU activation function."""
    return np.maximum(0, x)


def relu_derivative(x: ndarray) -> ndarray:
    """Compute the derivative of the ReLU activation function."""
    return np.where(x > 0, 1, 0)


def sigmoid(x: ndarray) -> ndarray:
    """Compute the sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: ndarray) -> ndarray:
    """Compute the derivative of the sigmoid activation function."""
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x: ndarray) -> ndarray:
    """Compute the tanh activation function."""
    return np.tanh(x)


def tanh_derivative(x: ndarray) -> ndarray:
    """Compute the derivative of the tanh activation function."""
    return 1 - np.tanh(x) ** 2


def softmax(x: ndarray) -> ndarray:
    """Compute the softmax of vector x in a numerically stable way."""
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)


act_derivative_mapping = {
    relu: relu_derivative,
    sigmoid: sigmoid_derivative,
    tanh: tanh_derivative,
    softmax: None
}
