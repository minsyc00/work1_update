"""Reproducible theorem experiment for the minimal exact CROWN implementation."""

from __future__ import annotations

from fractions import Fraction
from typing import Mapping

from .bpc import solve_crown_bpc
from .exact_oracle import compare_joint_and_sequential
from .instances import build_shared_corridor_counterexample


def run_shared_corridor_proof_experiment(
    *,
    agent_count: int = 3,
    epsilon: float = 0.5,
    verify_bpc: bool = True,
) -> Mapping[str, object]:
    """Execute the constructive proof and optionally cross-check CROWN-BPC."""

    instance, assignment = build_shared_corridor_counterexample(agent_count, epsilon)
    comparison = dict(compare_joint_and_sequential(instance, assignment))
    expected_sequential = float(agent_count)
    expected_joint = 1.0 + epsilon
    sequential = comparison["sequential"]
    joint = comparison["joint"]
    if abs(sequential.makespan - expected_sequential) > 1.0e-9:
        raise AssertionError("sequential theorem construction did not serialize as expected")
    if abs(joint.makespan - expected_joint) > 1.0e-9:
        raise AssertionError("joint theorem construction did not attain the private-route bound")

    comparison.update(
        {
            "agent_count": agent_count,
            "epsilon": epsilon,
            "theoretical_ratio": agent_count / (1.0 + epsilon),
        }
    )
    if verify_bpc:
        # Decimal benchmark inputs are converted to a finite rational grid so
        # both 1 and 1 + epsilon are represented without conservative rounding.
        epsilon_fraction = Fraction(str(epsilon)).limit_denominator(10_000)
        time_step = 1.0 / epsilon_fraction.denominator
        bpc = solve_crown_bpc(
            instance,
            horizon=float(agent_count),
            time_step=time_step,
        )
        comparison["bpc"] = bpc
        comparison["bpc_matches_enum"] = (
            abs(bpc.makespan - joint.makespan) <= 1.0e-9
            and abs(bpc.total_energy - joint.total_energy) <= 1.0e-9
        )
    return comparison
