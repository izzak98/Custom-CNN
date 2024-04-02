"""Module for utility functions."""
import cupy as np
from cupy import ndarray, float32


def convert_to_array(x) -> ndarray:
    """
    Converts the input `x` to a NumPy array of type float32.

    Parameters:
    x (array-like): The input data to be converted.

    Returns:
    ndarray: The converted NumPy array.

    """
    if not isinstance(x, ndarray):
        x = np.array(x)
    return x.astype(float32)
