# 200x200 S-Polygon Map Run Summary

Run date: 2026-07-02

Map:
`maps/static_obstacle_map_200x200_mixed_obstacles_s_polygon/static_obstacle_map_200x200_mixed_obstacles_s_polygon.json`

Output directory:
`outputs/static_obstacle_map_200x200_mixed_obstacles_s_polygon_usv4_footprint4x2_rmin2/paper_style_region_tsp/`

Important note:
The default full run with `coverage_aware_merge` enabled was interrupted after staying in `region_coarsen_merge` for more than 8 minutes. The complete image/report run below was a controlled run with `enable_coverage_aware_merge=False`, while keeping the rest of the planning pipeline enabled.

## Key Images

- `00_map_and_static_obstacles.png`
- `02_free_space_regions.png`
- `04_selected_region_sweep_patterns.png`
- `08_final_region_tsp_coverage_path.png`
- `09_constraint_validation.png`
- `11_repeat_overlap_diagnostics.png`
- `12_performance_metric_dashboard.png`
- `13_cross_agent_ownership_overlap.png`

## Core Metrics

- Total coverage fraction: `0.982175`
- Cover-only coverage fraction: `0.940825`
- Target coverage: `0.99`
- Coverage status: `incomplete`
- Target coverage status: `incomplete`
- Cover-only target status: `incomplete`
- Coverage quality status: `incomplete`
- Region execution status: `incomplete`
- Feasible regions: `53`
- Main TSP executed regions: `51`
- Skipped region count: `2`
- Residual count: `101`
- Residual area ratio: `0.017825`

## Constraint Metrics

- Invalid path length: `0.0 m`
- Out-of-bounds segment count: `0`
- Obstacle collision segment count: `0`
- Kinematic infeasible segment count: `0`
- Dynamic infeasible segment count: `0`
- Max curvature: `0.5`
- Max heading jump: `0.772771`
- Max yaw rate: `0.5`

## Path Economy Metrics

- Total path length: `25886.89 m`
- Coverage path length: `15269.17 m`
- Turn/transit path length: `10617.72 m`
- Coverage length ratio: `0.589842`
- Transition length ratio: `0.410158`
- Repeat overlap length: `3895.42 m`
- Repeat overlap ratio: `0.150478`
- Repeat-to-transition ratio: `0.366879`
- Total turn angle: `3711.96 rad`
- Turn count: `486`
- Turn angle per coverage meter: `0.2431 rad/m`
- Agent load imbalance: `0.207379`

## Stage Timing

- obstacle_normalization: `0.000104 s`
- free_space_decomposition: `0.585292 s`
- region_coarsen_merge: `0.000006 s` in the controlled run
- coverage_pattern_generation: `397.934918 s`
- build_sweep_paths: `805.826007 s`
- region_graph_building: `0.010401 s`
- load_balancing_assignment: `0.002063 s`
- coverage_ownership_map: `0.564061 s`
- agent_0_region_tsp: `139.870498 s`
- agent_1_region_tsp: `102.988025 s`
- agent_2_region_tsp: `343.365617 s`
- agent_3_region_tsp: `328.253794 s`
- skipped_region_recovery: `565.446811 s`
- residual_backfill_cycle_1: `601.616387 s`
- residual_backfill_cycle_2: `0.879940 s`
- cover_only_residual_backfill_cycle_1: `32.656736 s`

## Main Shortcomings

1. Coverage is not yet complete. The total coverage is `0.982175`, and the stricter cover-only coverage is only `0.940825`.
2. Two small feasible regions were skipped: `large_region_47` and `large_region_50`. Their recovery failed mainly because of heading jump, kinematic infeasibility, and obstacle collision in connectors.
3. The selected sweep patterns are all axis-aligned. `oriented_pattern_count=0`, `axis_aligned_pattern_count=97`, and `selected_oriented_pattern_count=0`, so the oriented sweep logic did not help this map.
4. Transition and turn cost is still high. Turn/transit length accounts for about `41.0%` of total path length.
5. Repeat overlap remains significant: `3895.42 m`, mostly from turn/transit rather than cover lines. Cross-agent overlap by kind is `transit=1133.02 m`, `turn=1755.96 m`, `cover=15.42 m`.
6. U-turn validation is a major bottleneck. `uturn_cache_miss_count=1825` and `uturn_cache_hit_count=0`.
7. Residual planning is expensive and still not enough. The first residual cycle added 5 candidates after 1824 attempts, but later cycles found no feasible append.
8. The default coverage-aware merge needs a strict budget/progress guard before it can be used safely on this 200x200 map.

