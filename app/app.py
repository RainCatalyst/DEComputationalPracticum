import streamlit as st
import pandas as pd
import numpy as np
import config
from solver.solver import *


def run():
    st.title('Differential Equation Solver')
    st.latex(config.equation_latex)

    # Get integration parameters
    params = _get_solver_params()

    # Solve using given methods and compute errors
    with st.spinner("Computing solutions..."):
        try:
            solution_data, lte_data = _compute_solutions_and_lte(params)
        except SolverError as e:
            st.error(e)
            return

    st.subheader("Solutions")
    st.line_chart(solution_data)

    st.subheader("Local Truncation Errors (LTE)")
    st.line_chart(lte_data)

    N0 = st.number_input('N0', 1, 1000, 5)
    N = st.number_input('N', int(N0), 1000, params.number_of_steps + 5)
    
    with st.spinner("Computing errors..."):
        try:
            gte_data = _compute_gte(params, N0, N)
        except SolverError as e:
            st.error(e)
            return

    st.subheader("Global Truncation Errors (GTE)")
    st.line_chart(gte_data)


@st.cache(suppress_st_warning=True)
def _compute_solutions_and_lte(params: SolverParams):
    solve_space = params.get_space()
    # Solve using different methods
    exact_solution = \
        ExactSolver.solve(config.solution, config.coefficient, params)
    euler_solution, euler_lte = \
        solve_and_compute_lte(EulerSolver, config.equation, exact_solution, params)
    improved_euler_solution, improved_euler_lte = \
        solve_and_compute_lte(ImprovedEulerSolver, config.equation, exact_solution, params)
    rk_solution, rk_lte = \
        solve_and_compute_lte(RungeKuttaSolver, config.equation, exact_solution, params)  

    # Solutions and errors plot Dataframe
    solution_data = pd.DataFrame(
        {
            'exact': exact_solution,
            'euler': euler_solution,
            'improved_euler': improved_euler_solution,
            'runge_kutta_solution': rk_solution
        }, index=solve_space)
    error_data = pd.DataFrame(
        {
            'euler_error': euler_lte,
            'improved_euler_error': improved_euler_lte,
            'runge_kutta_error': rk_lte
        }, index=solve_space)
    
    return solution_data, error_data


@st.cache(suppress_st_warning=True)
def _compute_gte(params: SolverParams, N0: int, N: int):    
    euler_gte, improved_euler_gte, rk_gte = [], [], []

    for n in range(N0, N):
        n_params = SolverParams(params.initial_value, params.x_from, params.x_to, n)

        exact_solution = ExactSolver.solve(config.solution, config.coefficient, n_params)
        euler_gte.append(
            np.max(solve_and_compute_lte(EulerSolver, config.equation, exact_solution, n_params)[1])
        )
        improved_euler_gte.append(
            np.max(solve_and_compute_lte(ImprovedEulerSolver, config.equation, exact_solution, n_params)[1])
        )
        rk_gte.append(
            np.max(solve_and_compute_lte(ImprovedEulerSolver, config.equation, exact_solution, n_params)[1])
        )
    
    # GTE Plot Dataframe
    gte_data = pd.DataFrame(
        {
            'euler_gte': euler_gte,
            'improved_euler_gte': improved_euler_gte,
            'runge_kutta_gte': rk_gte
        }, index=range(N0, N))

    return gte_data


def _get_solver_params() -> SolverParams:
    initial_value = st.number_input('Initial value', 0, 10, 2)
    x_from = st.number_input('X From', 0, 10, 1)
    x_to = st.number_input('X To', x_from, 100, 5)
    number_of_steps = st.number_input('Steps', 0, 1000, 50)
    return SolverParams(initial_value, x_from, x_to, number_of_steps)
