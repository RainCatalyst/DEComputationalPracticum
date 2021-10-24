import numpy as np

equation_latex = '\\huge f(x, y) = -\\frac{-y^2}{3} - \\frac{2}{3x^2}'
solution_latex = 'y(x) = e^{-2x}(\\frac{x^4}{4} + 1)'


def equation(x: float, y: float) -> float:
    return -np.power(y, 2) / 3 - 2 / (3 * np.power(x, 2))


def coefficient(x0: float, y0: float) -> float:
    return (np.power(x0, 2) * y0 - 2 * x0) / (np.power(x0, 2 / 3) * (1 - x0 * y0))


def solution(x: float, c: float) -> float:
    return 1 / (c * np.power(x, 2/3) + x) + 1 / x
