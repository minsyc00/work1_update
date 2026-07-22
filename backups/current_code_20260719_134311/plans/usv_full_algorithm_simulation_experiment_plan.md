# 全算法过程仿真实验增强计划

## Summary

本计划新增“规划层算法过程仿真实验”能力。实验不再只展示最终路径，而是按真实论文融合路径规划算法的执行顺序，逐阶段保存地图读入、障碍膨胀、自由空间分解、候选覆盖模式、区域图、负载均衡、单艇 TSP-CPP 初始解、2-opt 后优化、pattern 选择、绕障连接和最终 tour 的图像与指标。

## 关键目标

- 严格复用当前真实算法模块，避免为了画图写一套伪流程。
- 新增实验入口 `run_planning_algorithm_experiment(...)`，显式按 pipeline 阶段运行。
- 新增 `AlgorithmExperimentTrace`，保存所有阶段中间数据、耗时和指标。
- 输出 `algorithm_steps/` 子目录，包含阶段 PNG、算法过程 GIF 和 `algorithm_experiment_report.json`。
- 保持 `PathPlanningLayer.plan_from_config(...)` 默认行为不变。

## 输出内容

默认输出目录：

```text
outputs/<map_id>_usv2_footprint4x2_rmin2/algorithm_steps/
```

核心产物：

- `00_map_and_static_obstacles.png`
- `01_obstacle_inflation.png`
- `02_sweep_lines_and_free_cells.png`
- `03_decomposition_valid_cells.png`
- `04_candidate_coverage_patterns.png`
- `05_region_graph_weights.png`
- `06_balanced_assignment.png`
- `07_agent_<id>_tsp_initial_order.png`
- `08_agent_<id>_pattern_selection.png`
- `09_agent_<id>_2opt_iterations.png`
- `10_obstacle_aware_connections.png`
- `11_final_single_usv_tsp_cpp_tours.png`
- `12_algorithm_process.gif`
- `algorithm_experiment_report.json`

## 验收

- 20x20 双障碍、2 艘无人艇默认实验能完整生成上述文件。
- 报告中包含每个阶段耗时、区域数量、候选模式数量、负载差、每艇 TSP 顺序、2-opt 改善记录、pattern 选择代价和最终目标函数。
- 全量单元测试继续通过。
