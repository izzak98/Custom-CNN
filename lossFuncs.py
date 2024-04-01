import cupy as np


def mean_squared_error(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))


def mean_squared_error_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size


def categorical_crossentropy(y_true, y_pred):
    pred_vect = np.zeros_like(y_pred)
    pred_vect[np.arange(y_true.size), y_true.astype(np.int_)] = 1
    return -np.sum(pred_vect * np.log(y_pred + 1e-9)) / y_true.size


def categorical_crossentropy_derivative(y_true, y_pred):
    pred_vect = np.zeros_like(y_pred)
    pred_vect[np.arange(y_true.size), y_true.astype(np.int_)] = 1
    return (y_pred - pred_vect) / y_true.size


loss_derivative_mapping = {
    mean_squared_error: mean_squared_error_derivative,
    categorical_crossentropy: categorical_crossentropy_derivative
}

if __name__ == "__main__":
    y_true = np.array([0, 1, 3])  # One-hot encoded true labels
    y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])  # Predictions
    loss = categorical_crossentropy(y_true, y_pred)
    print(loss)
    derivative = categorical_crossentropy_derivative(y_true, y_pred)
    print(derivative)
