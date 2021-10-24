import numpy as np

equation_latex = '\huge f(x, y) = e^{-2x}x^3 - 2y'
solution_latex = 'y(x) = e^{-2x}(\frac{x^4}{4} + 1)'


def equation(x, y):
    return np.power(x, 3) * np.exp(-2 * x) - 2 * y


def solution(x):
    return np.exp(-2 * x) * (0.25 * np.power(x, 4) + 1)
