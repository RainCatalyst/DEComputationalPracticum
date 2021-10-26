import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import config
from solver.solver import *
from .chart import df_to_chart


def run():
    st.set_page_config(layout="wide")
    st.title('Differential Equation Solver')
    st.latex(config.equation_latex)

    _, params_col, solutions_col, _ = st.columns([1, 1, 4, 1])
    _, _, lte_col, _ = st.columns([1, 1, 4, 1])
    _, gte_params_col, gte_col, _ = st.columns([1, 1, 4, 1])

    # Get integration parameters
    with params_col:
        params = _get_solver_params()
        st.markdown(f'Step size: **{params.get_step_size():.3f}**')

    # Solve using given methods and compute errors
    try:
        solution_data, lte_data = _compute_solutions_and_lte(config.methods, params)    
    except SolverError as e:
        solutions_col.error(e)
        return

    solutions_col.subheader("Solutions")
    solutions_col.altair_chart(df_to_chart(solution_data), use_container_width=True)

    lte_col.subheader("Local Truncation Errors (LTE)")
    lte_col.altair_chart(df_to_chart(lte_data, y_label='lte'), use_container_width=True)

    N0 = gte_params_col.number_input('N0', 1, 1000, 10)
    N = gte_params_col.number_input('N', int(N0), 1000, 100)
    
    try:
        gte_data = _compute_gte(config.methods, params, N0, N)
    except SolverError as e:
        gte_col.error(e)
        return

    gte_col.subheader("Global Truncation Errors (GTE)")
    gte_col.altair_chart(df_to_chart(gte_data, x_label='n_points', y_label='gte'), use_container_width=True)


@st.cache(suppress_st_warning=True)
def _compute_solutions_and_lte(methods, params: SolverParams):
    solve_space = params.get_space()

    # Solve using different methods
    solutions, errors = {}, {}
    exact_solution = ExactSolver.solve(config.solution, config.coefficient, params)
    for name, solver in methods.items():
        solutions[name], errors[name] = \
            solve_and_compute_lte(solver, config.equation, exact_solution, params)

    solutions['_exact'] = exact_solution
    solution_data = pd.DataFrame(solutions, index=solve_space)
    error_data = pd.DataFrame(errors, index=solve_space)
    
    return solution_data, error_data


@st.cache(suppress_st_warning=True)
def _compute_gte(methods, params: SolverParams, N0: int, N: int):    
    errors = {name: [] for name in methods}

    for n in range(N0, N):
        n_params = SolverParams(params.initial_value, params.x_from, params.x_to, n)
        exact_solution = ExactSolver.solve(config.solution, config.coefficient, n_params)
        for name, solver in methods.items():
            errors[name].append(np.max(solve_and_compute_lte(solver, config.equation, exact_solution, n_params)[1]))
    
    # GTE Plot Dataframe
    gte_data = pd.DataFrame(errors, index=range(N0, N))

    return gte_data


def _get_solver_params() -> SolverParams:
    initial_value = st.number_input('y0', value=config.default_params.initial_value, step=1.0)
    x_from = st.number_input('x0', min_value=0.0, value=config.default_params.x_from, step=1.0)
    x_to = st.number_input('X', min_value=x_from, value=config.default_params.x_to, step=1.0)
    number_of_points = st.number_input('N', min_value=2, max_value=1000, value=config.default_params.number_of_points)
    return SolverParams(initial_value, x_from, x_to, number_of_points)
