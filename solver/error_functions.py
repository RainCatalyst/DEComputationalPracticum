import numpy as np


def absolute_error(exact_values: np.ndarray, approx_values: np.ndarray) -> np.ndarray:
    """Compute the absolute difference between elements in the arrays"""
    return np.abs(exact_values - approx_values)
