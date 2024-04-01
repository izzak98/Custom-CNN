import cupy as np
from cupy import ndarray


def relu(x: ndarray) -> ndarray:
    return np.maximum(0, x)


def relu_derivative(x: ndarray) -> ndarray:
    return np.where(x > 0, 1, 0)


def sigmoid(x: ndarray) -> ndarray:
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: ndarray) -> ndarray:
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x: ndarray) -> ndarray:
    return np.tanh(x)


def tanh_derivative(x: ndarray) -> ndarray:
    return 1 - np.tanh(x) ** 2


def softmax(x: ndarray) -> ndarray:
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)


def softmax_cross_entropy_derivative(y_true, y_pred):
    pred_vect = np.zeros_like(y_pred)
    pred_vect[np.arange(y_true.size), y_true.astype(np.int_)] = 1
    return y_pred - pred_vect


act_derivative_mapping = {
    relu: relu_derivative,
    sigmoid: sigmoid_derivative,
    tanh: tanh_derivative,
    softmax: softmax_cross_entropy_derivative
}
