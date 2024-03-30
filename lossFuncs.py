import cupy as np


def mean_squared_error(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))


def mean_squared_error_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size


loss_derivative_mapping = {
    mean_squared_error: mean_squared_error_derivative
}
