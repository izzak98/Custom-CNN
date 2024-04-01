import cupy as np
from cupy import ndarray, float_


class Adam:
    def __init__(self, lr=0.01, b1=0.9, b2=0.99, e=1e-9) -> None:
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.e = e
        self.m = None
        self.v = None
        self.t = 0

    def init_moments(self, weights: list[float_]) -> None:
        self.m = [np.zeros_like(weight) if weight is not None else None for weight in weights]
        self.v = self.m.copy()

    def __call__(self,
                 weights: list[float_],
                 gradients: dict[str, ndarray]
                 ):
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
            if "W" in grad_key:
                grads = grads
            try:
                self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * grads
                self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * grads ** 2
            except Exception as e:
                self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * grads.T
                self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * grads.T ** 2

            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            if i % 2 == 0:
                weight -= self.lr * m_hat / (np.sqrt(v_hat) + self.e)
            else:
                weight -= (self.lr * m_hat / (np.sqrt(v_hat) + self.e)).reshape(-1)
            updated_weights[grad_key] = weight

        return updated_weights
