# 多无人艇覆盖路径规划项目当前功能与流程总结

更新时间：2026-06-21  
项目根目录：`D:\code\work1_update`

## 1. 项目目标

本项目面向多无人艇集群的全覆盖路径规划与闭环跟踪，目标是在矩形任务区内，考虑矩形覆盖足迹、静态障碍物、3-DOF 无人艇运动学/动力学代理、最小转弯半径、区域间路径可达性、多艇任务分配和动态避障控制，生成可被 NMPC 跟踪的多艇覆盖路径。

当前项目已经形成两条主要能力线：

1. 基础全局覆盖与执行框架：矩形任务区条带覆盖、连续任务分配、CBS/MAPF 资源调度、Dubins/Bezier 平滑、时间参数化轨迹、RVO/CBF/NMPC 局部控制和闭环仿真。
2. 静态障碍地图论文式路径规划层：地图 JSON 读入、障碍物膨胀、自由空间分解、复合自由区、区块内部往复扫描、区块间 TSP-CPP、ACO/FA3ACO 可选求解、路径可跟踪验证、全过程图片和报告输出。

目前工程已具备较完整的模块结构和可视化诊断能力，但在 `50m x 50m` 及以上复杂静态障碍地图上，仍存在大图运行耗时长、区域间可达性不足、残差补扫闭环不稳定和覆盖率不足等问题。

## 2. 代码结构概览

核心代码位于 `src\usv_swarm`。

| 路径 | 作用 |
| --- | --- |
| `schema.py` | 定义任务区、舰队、足迹、安全参数、3-DOF 状态、轨迹参考、规划结果等基础数据结构。 |
| `planning.py` | 基础全局覆盖链路：条带生成、连续分配、CBS/MAPF、平滑、时间参数化参考轨迹。 |
| `dubins.py` | Dubins 最短路径和最小转弯半径几何代理。 |
| `geometry.py` | 几何工具、角度归一化、矩形足迹覆盖、连通区域等。 |
| `nmpc.py` | 基于 CasADi 的真 NMPC 控制器。 |
| `control.py` | 3-DOF 欠驱动无人艇模型、RVO-like 首选速度、CBF/NMPC 控制步、覆盖状态跟踪。 |
| `simulation.py` | 多艇闭环仿真、动态障碍样例、动画渲染。 |
| `path_planning\types.py` | 路径规划层所有核心中间类型。 |
| `path_planning\map_loader.py` | 地图 JSON 读入，并组合舰队配置生成 `PlannerConfig + StaticObstacle[]`。 |
| `path_planning\obstacles.py` | 静态障碍归一化、圆/椭圆离散、多边形膨胀和碰撞检查。 |
| `path_planning\decomposition.py` | 自由空间分解、静态障碍感知 cell 生成、复合自由空间区域构建。 |
| `path_planning\patterns.py` | 区域内候选覆盖模式、扫描线、覆盖 pass 生成。 |
| `path_planning\smoothing.py` | Dubins/Bezier/A*/motion lattice 区域间连接和平滑，可跟踪性转换。 |
| `path_planning\dynamics_validation.py` | 区域间连接的边界、障碍、曲率、航向、速度、yaw-rate、控制裕度验证。 |
| `path_planning\assignment.py` | 区域负载均衡分配。 |
| `path_planning\tsp.py` | 普通路径规划层的单艇 TSP-CPP 求解。 |
| `path_planning\aco.py` | ACO/FA3ACO 区域级 TSP-CPP 求解器。 |
| `path_planning\resources.py` | 重复路径惩罚、跨艇覆盖所有权、稳定 resource id、共享资源时间窗指标。 |
| `path_planning\scheduling.py` | resource-window 调度，处理共享通道/窄通道时间冲突。 |
| `path_planning\residuals.py` | 覆盖残差检测。 |
| `path_planning\residual_planner.py` | 残差局部 TSP-CPP 分配和补扫入口。 |
| `path_planning\performance.py` | 覆盖率、路径长度、重复覆盖、负载等性能指标。 |
| `path_planning\visualization.py` | 普通路径规划诊断图、全过程实验图、GIF 输出。 |
| `path_planning\paper_style_experiment.py` | 当前最重要的论文式区块扫描 + 区域间 TSP 实验入口。 |

示例脚本位于 `examples`，其中最常用的是：

| 脚本 | 作用 |
| --- | --- |
| `run_paper_style_region_tsp_experiment.py` | 论文式静态障碍区块扫描 + 区域间 TSP 实验主入口。 |
| `run_full_algorithm_experiment.py` | 全算法过程分阶段实验入口。 |
| `static_obstacle_planning_visual_diagnostics.py` | 静态障碍规划可视化诊断入口。 |
| `closed_loop_simulation.py` | 闭环控制仿真入口。 |

地图资产位于 `maps`，输出结果位于 `outputs`，计划文档位于 `plans`。

## 3. 主要数据结构

当前路径规划层通过 `PathPlanningConfig` 管理算法参数，通过 `MultiAgentPathPlan` 输出多艇路径。

关键类型如下：

| 类型 | 含义 |
| --- | --- |
| `PathWaypoint` | 路径点，字段为 `x, y, psi, time, speed`。 |
| `PathSegmentSpec` | 路径段，字段包括 `kind, waypoints, curvature_max, length, path_source, metadata`。 |
| `AgentPathPlan` | 单艇路径，包含多个 `PathSegmentSpec`。 |
| `MultiAgentPathPlan` | 多艇路径总结果。 |
| `StaticObstacle` | 静态障碍物，支持 polygon、rectangle、circle、ellipse 归一化为多边形。 |
| `ObstacleField` | 原始障碍与膨胀障碍集合。 |
| `FreeSpaceCell` | 静态障碍扣除后的自由空间基础 cell。 |
| `DecomposedRegion` | 普通矩形/多边形覆盖区块。 |
| `CompositeFreeSpaceRegion` | 由多个自由 cell 组成的复合覆盖区块，`bounds` 只用于索引和显示，真实覆盖由 `member_cells` 决定。 |
| `CoveragePass` | 区块内部单条往复扫描线段。 |
| `RegionCoveragePattern` | 一个区块的一种完整覆盖模式，包括扫描方向、入口、出口、pass 列表和代价。 |
| `RegionSweepPath` | 论文式区块内部往复扫描路径。 |
| `RegionVisitNode` | 区域级 TSP 节点，代表一个完整区块覆盖模式，不代表单条扫描线端点。 |
| `BalancedAssignment` | 多艇负载均衡后的区域分配。 |
| `CoverageOwnershipMap` | 跨艇覆盖所有权图，用于惩罚穿过其他艇已分配区域。 |
| `SingleUsvTourPlan` | 单艇区域访问顺序、模式选择和最终路径段。 |
| `PathPlanningTrace` | 普通可视化诊断快照。 |
| `AlgorithmExperimentTrace` | 全算法过程实验快照。 |

## 4. 基础全局覆盖流程

基础入口为：

```python
plan_global_coverage(config) -> PlanningResult
```

基础流程如下：

1. `build_boustrophedon_strips(config)`：在矩形任务区内生成往复式覆盖条带。
2. `solve_contiguous_partition(config, strips)`：把连续条带块分配给多艘无人艇。
3. `build_path_requirements(config, assignments)`：生成每艘艇路径需求和时空资源需求。
4. `solve_cbs_mapf(config, requirements)`：通过 CBS/MAPF 风格的预约表处理多艇资源冲突。
5. `build_smoothed_paths(config, reservations)`：使用 Dubins/Bezier 生成满足转弯半径约束的平滑路径。
6. `build_time_parameterized_references(config, paths)`：生成 `TrajectoryReference`，包含 `time, x, y, psi, u_ref, r_ref`。

这条链路主要适用于无内部静态障碍或简单矩形覆盖场景，是后续静态障碍路径规划层的基础适配目标。

## 5. 静态障碍地图与输出目录规范

当前地图 JSON 不包含无人艇初始状态，只保存静态环境资产。无人艇数量、初始位姿、速度、转弯半径等由实验配置注入。

已建立的地图包括：

| 地图 | 说明 |
| --- | --- |
| `static_obstacle_map_10x10_rect_obstacle` | `10m x 10m`，一个矩形障碍物。 |
| `static_obstacle_map_15x15_rect_triangle_small` | `15m x 15m`，一个小矩形和一个小三角形障碍物。 |
| `static_obstacle_map_20x20_two_obstacles` | `20m x 20m`，两个障碍物，用于论文式过程实验。 |
| `static_obstacle_map_50x50_simple` | `50m x 50m`，四个静态障碍物，用于大图实验。 |

地图资产目录结构示例：

```text
maps\static_obstacle_map_50x50_simple\
  static_obstacle_map_50x50_simple.json
  static_obstacle_map_50x50_simple.png
  static_obstacle_map_50x50_simple.md
```

运行结果目录命名遵循：

```text
outputs\<map_id>_usv<N>_footprint<lf>x<wf>_rmin<Rmin>\
```

例如：

```text
outputs\static_obstacle_map_50x50_simple_usv3_footprint4x2_rmin1\
```

## 6. 论文式静态障碍路径规划流程

当前重点流程位于：

```python
run_paper_style_region_tsp_experiment(...)
```

其核心思想是严格区分“区块内部覆盖”和“区块之间排序”：

1. 区块内部：每个自由空间区块生成完整往复扫描覆盖路径。
2. 区块之间：每个区块作为一个 TSP 节点，只优化区域访问顺序和每区入口/出口模式。
3. 扫描线端点不是 TSP 节点，扫描线端点只用于展开最终覆盖路径。

当前论文式流程如下：

1. 地图读入：从 JSON 读取任务矩形、覆盖足迹、转弯半径和静态障碍。
2. 障碍归一化：rectangle、polygon、circle、ellipse 统一转换为多边形。
3. 障碍膨胀：按 `d_safe`、足迹裕度和配置额外膨胀量生成保守障碍。
4. 自由空间分解：从任务矩形中扣除膨胀障碍，生成 `FreeSpaceCell`。
5. 复合自由区构建：用 `CompositeFreeSpaceRegion` 把相邻自由 cell 合成为论文式大区块。
6. 覆盖模式生成：对每个区块生成多候选扫描轴、入口/出口、口袋尺度和反向模式。
7. 扫描线裁剪：复合区域内部扫描线与 `member_cells` 求交，障碍洞不当作自由区。
8. 内部 U-turn 验证：相邻扫描线之间使用 Dubins/Bezier/A*/motion lattice 尝试连接并验证。
9. 区域图构建：以可行覆盖区块为节点，生成邻接图和区域权重。
10. 负载均衡：将区域分配给多艇。
11. 覆盖所有权图：负载均衡后建立 `CoverageOwnershipMap`，惩罚跨艇穿越其他艇区域。
12. 单艇区域 TSP-CPP：每艘艇只对自己分配到的区域节点求访问顺序。
13. 区域间连接：上一区域出口到下一区域入口依次尝试 Dubins/Bezier、A* corridor、corridor 转可跟踪子段、motion lattice、heading repair。
14. 动态可行性验证：区域间边必须通过边界、障碍、曲率、航向连续、yaw-rate、yaw-acceleration、速度和控制裕度验证。
15. 重复路径惩罚：主区域 TSP、内部 U-turn 和 residual local TSP 中加入已覆盖通道重复惩罚。
16. resource-window 调度：对共享 corridor、窄通道、turn pocket、cover strip 生成稳定 `resource_id` 并做时间窗错峰。
17. 残差检测与补扫：检测自由空间漏扫区域，尝试转成 residual regions 并进入局部 TSP-CPP。
18. 可视化与报告：输出 PNG、GIF 和 JSON 指标。

## 7. 当前路径约束处理方式

当前路径生成和验证同时考虑几何约束与动力学可跟踪性。

已实现的约束包括：

1. 地图边界约束：路径采样点必须在任务矩形内。
2. 静态障碍约束：cover、turn、transit、Bezier、Dubins、A* converted、motion lattice 段均需避开膨胀障碍。
3. 最小转弯半径：曲率上界满足 `kappa <= 1 / Rmin + 1e-3`。
4. 航向连续：拒绝硬折角和 heading jump 过大的连接。
5. 速度与 yaw-rate：按配置检查最大速度、最大 yaw-rate、yaw acceleration。
6. 控制裕度：估计 thrust 和 yaw moment 需求。
7. 3-DOF rollout/NMPC 可跟踪代理：将几何连接进一步验证为可跟踪连接。

路径段的可行性状态在 metadata 中区分：

1. `collision_free`
2. `curvature_feasible`
3. `kinematic_feasible`
4. `dynamic_feasible`

最终 TSP 图和最终路径应只允许 `dynamic_feasible=true` 的区域间边进入。`raw_astar_corridor_edge` 只允许作为诊断信息，不能直接进入最终路径。

## 8. TSP-CPP 求解器

当前支持三类区域级 TSP-CPP 求解器：

| 求解器 | 参数值 | 说明 |
| --- | --- | --- |
| 确定性 beam/greedy/2-opt | `deterministic` | 默认求解器，适合调试和稳定复现。 |
| ACO | `aco` | 蚁群算法，节点为 `(region_id, pattern_id)`，每个 region 只访问一次。 |
| FA3ACO | `fa3aco` | 工程版分数阶自适应 ACO，包含分数阶历史记忆、自适应挥发率和 3-opt 后处理。 |

命令行参数包括：

```text
--tsp-solver deterministic|aco|fa3aco
--aco-ants <数量>
--aco-iterations <迭代次数>
--aco-seed <随机种子>
```

若用户显式选择 ACO/FA3ACO，但求解器无法找到完整可行 tour，系统会记录 `tsp_solver_status=failed`，并允许回退到 deterministic，同时在报告中写明 `effective_tsp_solver`。

## 9. 执行控制与闭环仿真

执行控制层位于 `control.py`、`nmpc.py` 和 `simulation.py`。

当前执行控制流程如下：

1. 规划层输出 `TrajectoryReference`。
2. `SwarmRuntime.control_step(...)` 读取当前艇状态和局部参考窗口。
3. 使用 3-DOF 模型作为在线预测模型。
4. 使用 RVO-like 方式生成短时域首选速度。
5. CasADi NMPC 基于 3-DOF 预测模型，综合跟踪误差、速度误差、控制代价、输入变化率、边界约束、艇间 CBF 约束、动态障碍 CBF 约束求解控制量。
6. 若 NMPC 不可行，则降级到安全保持/低速停车模式。
7. `CoverageTracker` 使用矩形足迹更新覆盖状态并检测 residual。

当前 NMPC 是真 CasADi/Ipopt 求解器，不是占位模拟器。3-DOF 状态为：

```text
x3 = [x, y, psi, u, v, r]
```

控制量为：

```text
u3 = [T, N]
```

当前文档聚焦 3-DOF 无人艇模型：离线路径规划使用 Dubins/Bezier/motion lattice 作为 3-DOF 可跟踪几何代理，在线执行层使用 3-DOF NMPC 进行轨迹跟踪和安全避障。

## 10. 可视化与报告输出

论文式实验会在输出目录下生成：

| 文件 | 含义 |
| --- | --- |
| `00_map_and_static_obstacles.png` | 原始地图、障碍和无人艇初始状态。 |
| `01_obstacle_inflation.png` | 原始障碍与膨胀障碍对比。 |
| `02_free_space_regions.png` | 自由空间 cell、复合 region 和 region id。 |
| `03_feasible_region_sweep_modes.png` | 可行 sweep region。 |
| `04_region_sweep_patterns.png` | 每个区块内部候选往复扫描线。 |
| `04_selected_region_sweep_patterns.png` | 被最终选择的覆盖模式。 |
| `05_region_tsp_nodes.png` | 区域级 TSP 节点。 |
| `06_agent_region_tsp_order.png` | 每艘艇的区域访问顺序。 |
| `07_agent_sweep_endpoints.png` | 区块内部扫描线端点。 |
| `08_final_region_tsp_coverage_path.png` | 所有无人艇最终路径。 |
| `09_constraint_validation.png` | 边界、障碍、曲率、动力学约束检查。 |
| `10_shared_resource_timeline.png` | 共享资源时间窗，用于区分空间重合与真正时窗冲突。 |
| `11_repeat_overlap_diagnostics.png` | 重复路径诊断。 |
| `12_performance_metric_dashboard.png` | 路径长度、覆盖率、重复率、负载等指标仪表盘。 |
| `13_cross_agent_ownership_overlap.png` | 跨艇覆盖所有权和跨艇 overlap 诊断。 |
| `paper_style_region_tsp_report.json` | 完整 JSON 报告。 |

完整算法过程实验还会在 `algorithm_steps` 子目录输出从地图读入到 TSP 后优化的阶段图和 `algorithm_process.gif`。

## 11. 当前已完成功能

### 11.1 地图和静态障碍

已完成：

1. 地图 JSON 读入。
2. 地图资产与输出目录规范。
3. 支持 rectangle、polygon、circle、ellipse 障碍。
4. 圆/椭圆离散为多边形。
5. 障碍按安全裕度和足迹裕度膨胀。
6. 静态障碍碰撞检查。
7. 多个小型和中型静态障碍地图资产。

### 11.2 区域分解与复合自由区

已完成：

1. 静态障碍感知自由空间 cell 分解。
2. 基于相邻 cell 的复合自由区 `CompositeFreeSpaceRegion`。
3. 复合 region 的 `bounds` 与真实 `member_cells` 分离。
4. 扫描线与 member cells 求交，避免把障碍洞当作自由空间。
5. 大图模式下的候选预筛选和部分 region repair 入口。

### 11.3 区块内部覆盖

已完成：

1. 矩形覆盖足迹模型，使用 `lf, wf`。
2. 条带间距 `Delta = wf * (1 - rho)`。
3. 区块内部往复扫描。
4. 多扫描轴、多入口/出口、多 pocket scale 和反向模式。
5. 内部 U-turn Dubins/Bezier/A*/motion lattice 验证。
6. 覆盖模式代价中加入覆盖质量、路径长度、转角、重复路径和边界风险。

### 11.4 区域间路径规划

已完成：

1. 区域级 TSP 节点建模。
2. 确定性 TSP-CPP。
3. ACO/FA3ACO 可选 TSP-CPP。
4. 2-opt/3-opt 后优化接口。
5. Dubins/Bezier 区域间连接。
6. A* corridor 绕障搜索。
7. A* corridor 转 Dubins/Bezier/fillet/motion lattice 可跟踪子段。
8. motion lattice heading repair。
9. 严格验证失败边并记录 `infeasible_edges`。

### 11.5 多艇协同

已完成：

1. 多艇区域负载均衡。
2. 每艇单独区域 TSP-CPP。
3. 跨艇 coverage ownership map。
4. 穿过其他艇 owned 区域的软惩罚。
5. 重复通道软惩罚。
6. 稳定 resource id。
7. resource-window 调度，能够区分“空间重合但不同时间”和“真正资源时窗冲突”。

### 11.6 残差补扫

已完成：

1. 覆盖状态栅格。
2. 自由空间残差检测。
3. residual region 生成。
4. residual local TSP-CPP 入口。
5. residual 连接代价、重复路径惩罚和调度接口。

需要注意：残差补扫在小图和构造测试中已有能力，但在 `50m x 50m` 大图中还没有稳定把大量 skipped region 和 residual 全部补回。

### 11.7 控制与仿真

已完成：

1. 3-DOF 欠驱动水面艇模型。
2. CasADi NMPC。
3. CBF 风格艇间、边界、动态障碍安全约束。
4. RVO-like 首选速度。
5. NMPC 不可行时安全保持。
6. 闭环仿真和 GIF 动画。

### 11.8 测试

当前 `tests\test_path_planning_layer.py` 已覆盖较多关键能力，包括：

1. 路径规划层注册和 fallback。
2. 数据类型实例化。
3. 覆盖模型和 residual。
4. 区域分解、凹区域接口、复合区域。
5. Dubins 可行性。
6. 负载均衡。
7. turn-aware A*。
8. TSP-CPP。
9. ACO/FA3ACO 可复现性和后优化。
10. 静态障碍归一化和膨胀。
11. 地图 JSON 读入。
12. A* corridor 可跟踪化。
13. 动态验证拒绝硬折角。
14. resource-window 调度。
15. 重复路径惩罚。
16. 跨艇覆盖所有权惩罚。
17. 性能指标。
18. 可视化输出。
19. 论文式 region TSP 不把扫描端点当 TSP 节点。
20. 大图预筛选和合并安全性。

## 12. 当前典型运行命令

### 12.1 运行 15x15 小图论文式实验

```powershell
D:\anaconda3\envs\pytorch_gpu\python.exe examples\run_paper_style_region_tsp_experiment.py `
  --map maps\static_obstacle_map_15x15_rect_triangle_small\static_obstacle_map_15x15_rect_triangle_small.json `
  --rmin 0.5 `
  --usv-count 2 `
  --tsp-solver deterministic `
  --target-coverage 0.99 `
  --performance-profile balanced `
  --monitor-stages `
  --dpi 120
```

### 12.2 运行 50x50 大图论文式实验

```powershell
D:\anaconda3\envs\pytorch_gpu\python.exe examples\run_paper_style_region_tsp_experiment.py `
  --map maps\static_obstacle_map_50x50_simple\static_obstacle_map_50x50_simple.json `
  --rmin 1.0 `
  --usv-count 3 `
  --tsp-solver fa3aco `
  --aco-ants 40 `
  --aco-iterations 100 `
  --aco-seed 1234 `
  --target-coverage 0.99 `
  --performance-profile balanced `
  --monitor-stages `
  --dpi 120
```

### 12.3 快速报告模式

```powershell
D:\anaconda3\envs\pytorch_gpu\python.exe examples\run_paper_style_region_tsp_experiment.py `
  --map maps\static_obstacle_map_50x50_simple\static_obstacle_map_50x50_simple.json `
  --rmin 1.0 `
  --usv-count 3 `
  --tsp-solver deterministic `
  --monitor-stages `
  --no-render
```

### 12.4 运行单元测试

```powershell
D:\anaconda3\envs\pytorch_gpu\python.exe -m unittest discover -s tests -v
```

## 13. 当前 50x50 大图实验状态

最近一次完整目标配置为：

```text
map = static_obstacle_map_50x50_simple
USV = 3
footprint = 4m x 2m
Rmin = 1.0m
tsp_solver = fa3aco
target_coverage = 0.99
```

完整渲染版运行在 30 分钟内未完成，主要卡在：

| 阶段 | 观察到的耗时 |
| --- | --- |
| `build_sweep_paths` | 约 `1070s` |
| `composite_split_repair` | 约 `621s` |

为了产出图片，曾运行过一次降配诊断渲染版，输出目录为：

```text
outputs\static_obstacle_map_50x50_simple_usv3_footprint4x2_rmin1\paper_style_region_tsp
```

该诊断版结果：

| 指标 | 数值 |
| --- | --- |
| `coverage_fraction` | `0.5788` |
| `cover_only_coverage_fraction` | `0.4560` |
| `transit_assisted_coverage_fraction` | `0.5788` |
| `region_count` | `87` |
| `feasible_region_count` | `82` |
| `skipped_region_count` | `63` |
| `reachable_region_count` | `19` |
| `residual_count` | `9` |
| `total_length` | `715.99m` |
| `coverage_length` | `345.81m` |
| `transition_length` | `370.18m` |
| `repeat_overlap_length` | `64.56m` |
| `out_of_bounds` | `0` |
| `obstacle_collision` | `0` |
| `kinematic_infeasible` | `0` |
| `dynamic_infeasible` | `0` |

结论：

当前 50x50 失败的主要原因不是最终路径违反 3-DOF、Rmin、边界或障碍约束，而是大量可行 sweep region 在区域间 TSP/连接可达性阶段被跳过，导致覆盖不足。

## 14. 当前主要缺陷与待完善问题

### 14.1 大地图运行时间过长

问题：

1. `build_sweep_paths` 对复合 region 的候选模式做了大量重验证。
2. U-turn 验证缓存命中率低，最近诊断中 `uturn_cache_hit_count=2`、`uturn_cache_miss_count=543`。
3. composite split repair 会重新生成和验证大量候选，导致运行时间爆炸。
4. Agent 0 的区域 TSP 在降配诊断下仍耗时约 `616s`。

需要完善：

1. 更强的 U-turn cache key 归一化和复用。
2. 分层验证，先轻量几何过滤，再做动态验证。
3. composite split repair 只修复失败局部，而不是重新验证整批候选。
4. 大图 TSP 先构建稀疏 reachability graph，再限制候选边。

### 14.2 50x50 覆盖不完整

问题：

1. 有 82 个可行 sweep region，但最终只访问约 19 个。
2. Agent 0 分配 33 个 region，只访问 9 个。
3. Agent 2 分配 47 个 region，只访问 8 个。
4. 大量 region 因区域间连接失败被 skipped。

需要完善：

1. TSP 前建立 `region reachability graph`。
2. 负载均衡要基于可达连通分量，而不是只看区域权重。
3. 分配后若某艇 region 不连通，应迁移不可达 component 或插入桥接 region。
4. TSP 不应静默跳过大量 region，应转入 skipped-region recovery 或 residual local TSP。

### 14.3 区域间连接仍不够稳健

问题：

1. 部分区域间边失败于 `obstacle_collision`、`kinematic_infeasible`、`astar_corridor_conversion_failed`。
2. A* corridor 虽已实现可跟踪化转换，但复杂障碍/窄通道下仍有失败。
3. motion lattice heading repair 有成功案例，但不是所有 corridor 都能修复。

需要完善：

1. 为区域入口/出口增加 inward-offset、side-entry、反向 entry 和 boundary-safe entry。
2. corridor-to-trackable 转换需要更稳定的 fillet、heading adapter 和局部 lattice window。
3. 对过窄 corridor 应明确报告几何不可达，而不是让 TSP 后续跳过。
4. 可行连接图应缓存 region-pair 验证结果。

### 14.4 复合区域内部 U-turn 和覆盖裁剪仍有边界问题

问题：

1. 少量 composite region 在构建 sweep path 时失败。
2. 失败原因包括 `cover_invalid:obstacle_collision`、`uturn_invalid:obstacle_collision`、`uturn_invalid:out_of_bounds`。
3. 复合 region 过大或形状狭长时，内部 U-turn 可能无法在区域内完成。

需要完善：

1. 失败 region 应局部拆分为更小 composite subregions。
2. 对障碍边缘附近 pass 应使用更精确的 footprint-band 裁剪。
3. 入口/出口 pocket scale 应结合障碍 clearance 自适应。
4. 若区块内部无法完成 U-turn，应优先换扫描轴或拆分，而不是简单判整个 region 不可行。

### 14.5 残差补扫大图闭环不稳定

问题：

1. residual local TSP 已有接口，但大图中还没有稳定把 skipped region 和 residual 全部补回。
2. 诊断运行中为了产图关闭了 residual backfill，因此 residual_count 仍为 9。
3. 如果前置 TSP 已跳过大量 region，残差补扫会面临同样的区域间连接可达性问题。

需要完善：

1. residual region 使用和主 region 相同的多入口、多连接、动态验证流程。
2. residual local TSP 需要复用 reachability graph 和 resource scheduling。
3. skipped region recovery 与 residual backfill 应统一，避免两个补救机制互相割裂。

### 14.6 MAPF/CBS 仍是 resource-window hook，不是完整 CBS

问题：

当前已经能用 stable `resource_id` 做共享 corridor、窄通道、turn pocket 和 cover strip 的时间窗错峰，但这更接近 resource-window 调度 hook，还不是完整的 CBS 搜索树。

需要完善：

1. 将路径段抽象为 `PathRequirement`。
2. 对冲突生成 CBS constraints。
3. 对局部路径进行 wait/hold/replan。
4. 与区域 TSP 和 residual backfill 联动。

### 14.7 动态障碍仍未进入离线路径规划层

问题：

动态障碍当前主要由 CBF/RVO/NMPC 局部层处理，离线路径规划层只保留了接口和数据类型，没有把动态障碍转成时空约束参与全局规划。

需要完善：

1. 引入 time-expanded A* 或时空 MAPF。
2. 将动态障碍预测轨迹转成时空禁行区。
3. 将全局时间窗调度与局部 CBF/RVO/NMPC 联动。

### 14.8 3-DOF 动力学验证仍需进一步贴近真实执行

问题：

当前区域间路径已经检查边界、障碍、曲率、航向连续、速度、yaw-rate、yaw-acceleration 和控制裕度，但仍属于 3-DOF 工程代理验证。部分复杂 corridor 虽然几何上可行，仍可能因为短距离急转、heading adapter 不稳定或局部速度参数化不合理而被拒绝。

需要完善：

1. 强化 3-DOF rollout 验证，让区域间连接在进入 TSP 图前先完成更严格的可跟踪性检查。
2. 将速度剖面、yaw-rate 限幅和控制裕度更紧密地纳入 Dubins/Bezier/motion lattice 连接生成。
3. 对失败边输出更细的 3-DOF 原因分类，例如 `heading_jump`、`yaw_rate_exceeded`、`control_margin_exceeded`、`retime_failed`。
4. 增加 3-DOF NMPC 短时域跟踪验证，用于筛掉几何可行但控制层难以跟踪的区域间连接。

### 14.9 可视化和报告需要继续区分运行配置

问题：

完整高目标运行、降配诊断运行、无渲染快速运行可能写入相同实验目录，容易误读图片和报告。

需要完善：

1. 输出目录或报告中明确标记 `diagnostic_fast`、`full_target`、`no_render`。
2. 不同配置自动生成不同 run id。
3. 图片标题中显示 `target_coverage`、solver、候选限制、residual 是否启用。

## 15. 后续优先级建议

建议下一阶段按以下顺序推进：

1. 优先修复大图性能：U-turn cache、候选预筛选、局部 split repair、region-pair connector cache。
2. 在 TSP 前建立可行区域连接图：先知道哪些 region 彼此可达，再做负载均衡和 TSP。
3. 修复大图分配：按 reachability component 分配，避免某艘艇拿到大量不可达 region。
4. 稳定 A* corridor 可跟踪化：提升 fillet、heading adapter、motion lattice repair 成功率。
5. 将 skipped region recovery 和 residual local TSP 合并成统一补扫闭环。
6. 完整接入 resource-window/CBS：把共享通道时间冲突从诊断变成正式调度约束。
7. 再做 ACO/FA3ACO 大图性能优化：先保证可达图和边可行，再让智能优化算法参与排序。
8. 最后扩展动态障碍离线路径规划和更严格的 3-DOF NMPC 可跟踪验证。

## 16. 当前结论

当前项目已经完成了一个相当完整的多无人艇覆盖路径规划工程框架：从地图资产、静态障碍处理、区域分解、论文式区块扫描、区域间 TSP-CPP、ACO/FA3ACO、Dubins/Bezier/A*/motion lattice 可跟踪连接、3-DOF 动态验证、重复路径惩罚、跨艇覆盖所有权、资源时间窗调度，到 NMPC 闭环控制和全过程可视化，都已经有可运行代码和测试覆盖。

但项目还没有达到“复杂 50x50 静态障碍地图稳定全覆盖”的最终状态。当前大图主要瓶颈是运行时间和区域间可达性，而不是最终路径约束违规。下一步应重点把自由空间复合区块、可达连接图、负载均衡、区域 TSP、残差补扫和调度串成一个真正闭环的高覆盖率流程。
