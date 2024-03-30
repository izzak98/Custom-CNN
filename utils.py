from typing import Callable
import cupy as np
from cupy import ndarray, float32
import sympy as sp


def get_derivative(func: Callable, var: str, n_inputs: int) -> Callable:
    symbols = sp.symbols(f'x0:{n_inputs}')
    var_symbol = symbols[int(var.replace('x', ''))]
    func_symbolic = func(*symbols)
    derivative = sp.diff(func_symbolic, var_symbol)
    derivative_func = sp.lambdify(symbols, derivative, 'numpy')

    return derivative_func


def convert_to_array(x) -> ndarray:
    if not isinstance(x, ndarray):
        x = np.array(x)
    return x.astype(float32)
