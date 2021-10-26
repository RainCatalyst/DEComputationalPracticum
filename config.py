from solver.solver import *

methods = {
    'euler': EulerSolver,
    'improved_euler': ImprovedEulerSolver,
    'runge_kutta': RungeKuttaSolver
}

default_initial_value = 2.0
default_x0 = 1.0
default_X = 5.0
default_N = 15

equation_latex = '\\huge f(x, y) = -\\frac{-y^2}{3} - \\frac{2}{3x^2}'
solution_latex = 'y(x) = e^{-2x}(\\frac{x^4}{4} + 1)'


def equation(x: float, y: float) -> float:
    return -y * y / 3 - 2 / (3 * x * x)


def coefficient(x0: float, y0: float) -> float:
    return ((x0**(1/3)) * (2 - x0 * y0)) / (x0 * y0 - 1)


def solution(x: float, c: float) -> float:
    return 1 / x + 1 / (x + (x ** (2 / 3)) * c)
