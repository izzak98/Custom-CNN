"""A module for the Adam optimizer."""
import cupy as np
from cupy import ndarray, float_


class Adam:
    """Adam optimizer for training neural networks."""

    def __init__(self, lr=0.01, b1=0.9, b2=0.99, e=1e-9) -> None:
        """
        Initializes the Optimizer object.

        Args:
            lr (float): The learning rate for the optimizer. Default is 0.01.
            b1 (float): The exponential decay rate for the first moment estimates. Default is 0.9.
            b2 (float): The exponential decay rate for the second moment estimates. Default is 0.99.
            e (float): A small value to prevent division by zero. Default is 1e-9.
        """
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.e = e
        self.m = None
        self.v = None
        self.t = 0

    def init_moments(self, weights: list[float_]) -> None:
        """
        Initializes the moments for the optimizer.

        Args:
            weights (list[ndarray]): A list of weights.

        Returns:
            None
        """
        self.m = [np.zeros_like(weight) if weight is not None else None for weight in weights]
        self.v = self.m.copy()

    def __call__(self,
                 weights: list[ndarray],
                 gradients: dict[str, ndarray]
                 ) -> dict[str, ndarray]:
        """
        Update the weights using the Adam optimizer.

        Args:
            weights (list[ndarray]): The current weights of the model.
            gradients (dict[str, ndarray]): The gradients of the weights.

        Returns:
            dict[str, ndarray]: The updated weights after applying the Adam optimizer.
        """
        if self.m is None or self.v is None:
            self.init_moments(weights)

        assert self.m is not None
        assert self.v is not None

        self.t += 1
        updated_weights = {}

        for i, weight in enumerate(weights):

            if i % 2 == 0:
                grad_key = f'W{i // 2}'
            else:
                grad_key = f'b{i // 2}'

            if weight is None:
                updated_weights[grad_key] = None
                continue
            grads = gradients[grad_key]
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * grads
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * grads ** 2

            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            if i % 2 == 0:
                weight -= self.lr * m_hat / (np.sqrt(v_hat) + self.e)
                weight -= self.lr * m_hat / (np.sqrt(v_hat) + self.e)
            else:
                weight -= (self.lr * m_hat / (np.sqrt(v_hat) + self.e)).reshape(-1)
                weight -= (self.lr * m_hat / (np.sqrt(v_hat) + self.e)).reshape(-1)
            updated_weights[grad_key] = weight

        return updated_weights
