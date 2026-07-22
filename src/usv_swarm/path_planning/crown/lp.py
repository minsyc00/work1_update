"""Small dependency-free two-phase simplex used by the exact CROWN master.

This is not intended to replace an industrial LP solver.  It supports exactly
the form needed by the minimal CROWN-BPC implementation: non-negative
variables, equality rows, and ``<=`` rows with non-negative right-hand sides.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


_TOL = 1.0e-9


@dataclass(frozen=True)
class LinearProgramSolution:
    values: np.ndarray
    objective: float
    equality_duals: np.ndarray
    inequality_duals: np.ndarray
    iterations: int


class LinearProgramInfeasible(RuntimeError):
    pass


class LinearProgramUnbounded(RuntimeError):
    pass


def _pivot(
    tableau: np.ndarray,
    basis: list[int],
    row: int,
    column: int,
) -> None:
    pivot_value = tableau[row, column]
    tableau[row, :] /= pivot_value
    for other_row in range(tableau.shape[0]):
        if other_row == row:
            continue
        multiplier = tableau[other_row, column]
        if abs(multiplier) > _TOL:
            tableau[other_row, :] -= multiplier * tableau[row, :]
    basis[row] = column


def _simplex_maximize(
    tableau: np.ndarray,
    basis: list[int],
    objective: np.ndarray,
    *,
    max_iterations: int,
) -> Tuple[np.ndarray, int]:
    objective_row = np.concatenate((-objective.astype(float, copy=True), [0.0]))
    for row, basic_column in enumerate(basis):
        coefficient = objective[basic_column]
        if abs(coefficient) > _TOL:
            objective_row += coefficient * tableau[row, :]

    iterations = 0
    while True:
        negative = np.flatnonzero(objective_row[:-1] < -_TOL)
        if negative.size == 0:
            return objective_row, iterations
        entering = int(negative[0])  # Bland's rule prevents cycling.
        candidates = []
        for row in range(tableau.shape[0]):
            coefficient = tableau[row, entering]
            if coefficient > _TOL:
                candidates.append((tableau[row, -1] / coefficient, basis[row], row))
        if not candidates:
            raise LinearProgramUnbounded("LP relaxation is unbounded")
        _, _, leaving_row = min(candidates)
        _pivot(tableau, basis, leaving_row, entering)
        multiplier = objective_row[entering]
        objective_row -= multiplier * tableau[leaving_row, :]
        iterations += 1
        if iterations > max_iterations:
            raise RuntimeError("simplex exceeded max_iterations")


def solve_linear_program(
    objective: Sequence[float],
    equality_matrix: Sequence[Sequence[float]],
    equality_rhs: Sequence[float],
    inequality_matrix: Sequence[Sequence[float]],
    inequality_rhs: Sequence[float],
    *,
    max_iterations: int = 100_000,
) -> LinearProgramSolution:
    """Minimize a linear objective under equality/``<=`` constraints."""

    c = np.asarray(objective, dtype=float)
    if c.ndim != 1:
        raise ValueError("objective must be one-dimensional")
    variable_count = c.size

    def matrix_or_empty(rows: Sequence[Sequence[float]]) -> np.ndarray:
        array = np.asarray(rows, dtype=float)
        if array.size == 0:
            return np.zeros((0, variable_count), dtype=float)
        return array.reshape((-1, variable_count))

    a_eq = matrix_or_empty(equality_matrix)
    b_eq = np.asarray(equality_rhs, dtype=float)
    a_ub = matrix_or_empty(inequality_matrix)
    b_ub = np.asarray(inequality_rhs, dtype=float)
    if a_eq.shape[0] != b_eq.size or a_ub.shape[0] != b_ub.size:
        raise ValueError("constraint matrix and RHS dimensions do not match")
    if not (
        np.all(np.isfinite(c))
        and np.all(np.isfinite(a_eq))
        and np.all(np.isfinite(b_eq))
        and np.all(np.isfinite(a_ub))
        and np.all(np.isfinite(b_ub))
    ):
        raise ValueError("LP data must be finite")
    if np.any(b_ub < -_TOL):
        raise ValueError("this simplex implementation requires non-negative <= RHS")
    b_ub[np.abs(b_ub) <= _TOL] = 0.0

    # Equality rows may be multiplied without changing their type.
    for row in range(b_eq.size):
        if b_eq[row] < -_TOL:
            a_eq[row, :] *= -1.0
            b_eq[row] *= -1.0

    equality_count = a_eq.shape[0]
    inequality_count = a_ub.shape[0]
    row_count = equality_count + inequality_count
    if row_count == 0:
        if np.any(c < -_TOL):
            raise LinearProgramUnbounded("unconstrained objective is unbounded")
        return LinearProgramSolution(
            values=np.zeros(variable_count),
            objective=0.0,
            equality_duals=np.zeros(0),
            inequality_duals=np.zeros(0),
            iterations=0,
        )

    slack_count = inequality_count
    artificial_count = equality_count
    non_artificial_count = variable_count + slack_count
    total_columns = non_artificial_count + artificial_count
    standard = np.zeros((row_count, non_artificial_count), dtype=float)
    standard[:equality_count, :variable_count] = a_eq
    standard[equality_count:, :variable_count] = a_ub
    if slack_count:
        standard[
            equality_count:,
            variable_count : variable_count + slack_count,
        ] = np.eye(slack_count)

    phase_one_matrix = np.zeros((row_count, total_columns), dtype=float)
    phase_one_matrix[:, :non_artificial_count] = standard
    if artificial_count:
        phase_one_matrix[
            :equality_count,
            non_artificial_count:,
        ] = np.eye(artificial_count)
    rhs = np.concatenate((b_eq, b_ub))
    tableau = np.column_stack((phase_one_matrix, rhs))
    basis = [
        non_artificial_count + row for row in range(equality_count)
    ] + [
        variable_count + row for row in range(inequality_count)
    ]

    phase_one_objective = np.zeros(total_columns)
    phase_one_objective[non_artificial_count:] = -1.0
    phase_one_row, phase_one_iterations = _simplex_maximize(
        tableau,
        basis,
        phase_one_objective,
        max_iterations=max_iterations,
    )
    if phase_one_row[-1] < -_TOL:
        raise LinearProgramInfeasible("LP constraints are infeasible")

    active_rows = list(range(row_count))
    redundant_rows = []
    for row in range(tableau.shape[0]):
        if basis[row] < non_artificial_count:
            continue
        entering = next(
            (
                column
                for column in range(non_artificial_count)
                if column not in basis and abs(tableau[row, column]) > _TOL
            ),
            None,
        )
        if entering is not None:
            _pivot(tableau, basis, row, entering)
        elif abs(tableau[row, -1]) <= _TOL:
            redundant_rows.append(row)
        else:
            raise LinearProgramInfeasible("artificial variable remains positive")

    for row in reversed(redundant_rows):
        tableau = np.delete(tableau, row, axis=0)
        del basis[row]
        del active_rows[row]
    tableau = np.delete(
        tableau,
        np.s_[non_artificial_count:total_columns],
        axis=1,
    )

    minimization_cost = np.concatenate((c, np.zeros(slack_count)))
    phase_two_row, phase_two_iterations = _simplex_maximize(
        tableau,
        basis,
        -minimization_cost,
        max_iterations=max_iterations,
    )
    del phase_two_row

    standard_values = np.zeros(non_artificial_count)
    for row, basic_column in enumerate(basis):
        standard_values[basic_column] = max(0.0, tableau[row, -1])
    values = standard_values[:variable_count]
    objective_value = float(c @ values)

    active_standard = standard[np.asarray(active_rows), :]
    basis_matrix = active_standard[:, np.asarray(basis)]
    active_duals = np.linalg.solve(
        basis_matrix.T,
        minimization_cost[np.asarray(basis)],
    )
    all_duals = np.zeros(row_count)
    all_duals[np.asarray(active_rows)] = active_duals
    reduced_costs = minimization_cost - standard.T @ all_duals
    if np.min(reduced_costs) < -1.0e-6:
        raise RuntimeError("simplex returned a dual-infeasible basis")

    return LinearProgramSolution(
        values=values,
        objective=objective_value,
        equality_duals=all_duals[:equality_count],
        inequality_duals=all_duals[equality_count:],
        iterations=phase_one_iterations + phase_two_iterations,
    )
