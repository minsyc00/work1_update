"""Run the minimal exact CROWN counterexample and BPC cross-check."""

from __future__ import annotations

import argparse
import json

from usv_swarm.path_planning.crown import run_shared_corridor_proof_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=0.5)
    args = parser.parse_args()

    result = run_shared_corridor_proof_experiment(
        agent_count=args.agents,
        epsilon=args.epsilon,
        verify_bpc=True,
    )
    sequential = result["sequential"]
    joint = result["joint"]
    bpc = result["bpc"]
    report = {
        "agents": args.agents,
        "epsilon": args.epsilon,
        "sequential_exact_post": {
            "makespan": sequential.makespan,
            "energy": sequential.total_energy,
            "nominal_makespan": sequential.statistics["nominal_makespan"],
            "deconfliction_penalty": sequential.statistics["deconfliction_penalty"],
        },
        "joint_exact": {
            "makespan": joint.makespan,
            "energy": joint.total_energy,
        },
        "joint_gain": result["joint_gain"],
        "observed_ratio": result["joint_gain_ratio"],
        "theoretical_ratio": result["theoretical_ratio"],
        "strict_improvement": result["strict_improvement"],
        "bpc": {
            "makespan": bpc.makespan,
            "energy": bpc.total_energy,
            "lower_bound": bpc.lower_bound,
            "upper_bound": bpc.upper_bound,
            "gap": bpc.optimality_gap,
            "energy_lower_bound": bpc.energy_lower_bound,
            "energy_upper_bound": bpc.energy_upper_bound,
            "energy_gap": bpc.energy_optimality_gap,
            "generated_columns": bpc.generated_columns,
            "pricing_iterations": bpc.pricing_iterations,
            "branch_nodes": bpc.branch_nodes,
            "conflict_separation_rounds": bpc.conflict_separation_rounds,
        },
        "bpc_matches_enum": result["bpc_matches_enum"],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
