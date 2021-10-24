import numpy as np
from dataclasses import dataclass, replace
from .error_functions import absolute_error


class SolverError(Exception):
    pass


@dataclass
class SolverParams:
    """Class for storing integration parameters

    Args:
        initial_value (float): value of y(x_from)
        x_from (float): integration range start
        x_to (float): integration range end
        number_of_steps (int): number of points to integrate at

    """
    initial_value: float
    x_from: float
    x_to: float
    number_of_steps: int

    def get_step_size(self) -> float:
        """Computes step size for integration"""
        return (self.x_to - self.x_from) / self.number_of_steps

    def get_space(self, skip_first=False) -> np.ndarray:
        """Get an array of all points inside the params range"""
        if skip_first:
            return np.linspace(self.x_from, self.x_to, self.number_of_steps)[1:]
        return np.linspace(self.x_from, self.x_to, self.number_of_steps)


class Solver(object):
    @staticmethod
    def solve(equation, params: SolverParams) -> np.ndarray:
        """Solve the differential equation using params

        Args:
            equation (Callable): equation of type f(x, y)
            params (SolverParams): steps, initial value and range

        Returns:
            np.ndarray: approximated y values of the function
        """
        return np.array([])


class EulerSolver(Solver):
    @staticmethod
    def solve(equation, params: SolverParams) -> np.ndarray:
        """Solve the differential equation using Euler Method

        Args:
            equation (Callable): equation of type f(x, y)
            params (SolverParams): steps, initial value and range

        Returns:
            np.ndarray: approximated y values of the function
        """

        step_size = params.get_step_size()
        if step_size >= 1:
            raise SolverError(f"Solver step size {step_size} is >= 1")

        y = params.initial_value
        values = [y]
        for x in params.get_space(skip_first=True):
            y += step_size * equation(x, y)
            values.append(y)
        return np.array(values)


class ImprovedEulerSolver(Solver):
    @staticmethod
    def solve(equation, params: SolverParams) -> np.ndarray:
        """Solve the differential equation using the Improved Euler Method

        Args:
            equation (Callable): equation of type f(x, y)
            params (SolverParams): number of steps, initial value and range

        Returns:
            np.ndarray: approximated y values of the function
        """
        
        step_size = params.get_step_size()
        if step_size >= 1:
            raise SolverError(f"Solver step size {step_size} is >= 1")

        y = params.initial_value
        values = [y]
        for x in params.get_space(skip_first=True):
            slope1 = equation(x, y)
            slope2 = equation(x + step_size, y + step_size * slope1)

            y += 0.5 * step_size * (slope1 + slope2)
            values.append(y)
        return np.array(values)


class RungeKuttaSolver(Solver):
    @staticmethod
    def solve(equation, params: SolverParams) -> np.ndarray:
        """Solve the differential equation using the Runge-Kutta Method with 4 sample points

        Args:
            equation (Callable): equation of type f(x, y)
            params (SolverParams): number of steps, initial value and range

        Returns:
            np.ndarray: approximated y values of the function
        """
        step_size = params.get_step_size()
        if step_size >= 1:
            raise SolverError(f"Solver step size {step_size} is >= 1")

        y = params.initial_value
        values = [y]
        for x in params.get_space(skip_first=True):
            slope1 = equation(x, y)
            slope2 = equation(x + 0.5 * step_size, y + 0.5 * step_size * slope1)
            slope3 = equation(x + 0.5 * step_size, y + 0.5 * step_size * slope2)
            slope4 = equation(x + step_size, y + step_size * slope3)

            y += step_size / 6 * (slope1 + 2 * slope2 + 2 * slope3 + slope4)
            values.append(y)
        return np.array(values)


class ExactSolver(Solver):
    @staticmethod
    def solve(equation, coefficient_equation, params: SolverParams) -> np.ndarray:
        """Solve the differential equation using the provided analytical solution

        Args:
            equation (Callable): equation of type y(x, c)
            coefficient_equation (Callable): formula for computing the coefficient (of type c(x0, y0))
            params (SolverParams): number of steps, initial value and range

        Returns:
            np.ndarray: exact y values of the function
        """
        c = coefficient_equation(params.x_from, params.initial_value)
        return equation(params.get_space(), c)


def solve_and_compute_lte(solver: Solver, equation, exact_values: np.ndarray, params: SolverParams):
    """Solves the equation and computes absolute errors for exact solution"""
    approx_values = solver.solve(equation, params)
    errors = absolute_error(exact_values, approx_values)
    return approx_values, errors