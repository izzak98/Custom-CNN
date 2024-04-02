"""This module contains the loss functions and their derivatives."""
import cupy as np
from cupy import ndarray


def mean_squared_error(y_true, y_pred) -> ndarray:
    """Compute the mean squared error loss function."""
    return np.mean(np.square(y_true - y_pred))


def mean_squared_error_derivative(y_true, y_pred) -> ndarray:
    """Compute the derivative of the mean squared error loss function."""
    return 2 * (y_pred - y_true) / y_true.size


def sparse_categorical_crossentropy(y_true, y_pred_logits) -> ndarray:
    """Compute the sparse categorical crossentropy loss function."""
    n_samples = y_pred_logits.shape[0]
    y_true_one_hot = np.eye(y_pred_logits.shape[1])[y_true.astype(int)]
    loss = -np.sum(y_true_one_hot * np.log(y_pred_logits + 1e-15)) / n_samples
    return loss


def sparse_categorical_crossentropy_derivative(y_true, y_pred_logits) -> ndarray:
    """Compute the derivative of the sparse categorical crossentropy loss function."""
    n_samples = y_pred_logits.shape[0]
    y_true_one_hot = np.eye(y_pred_logits.shape[1])[y_true.astype(int)].reshape(y_pred_logits.shape)
    return (y_pred_logits - y_true_one_hot) / n_samples


loss_derivative_mapping = {
    mean_squared_error: mean_squared_error_derivative,
    sparse_categorical_crossentropy: sparse_categorical_crossentropy_derivative
}
