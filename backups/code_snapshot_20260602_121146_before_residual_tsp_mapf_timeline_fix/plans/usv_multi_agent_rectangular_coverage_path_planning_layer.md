# 多无人艇矩形覆盖路径规划层算法方案

## 摘要

基于 `articles` 中六篇论文，路径规划层采用“区域分解 + 负载均衡 + 转弯约束覆盖模式 + A* 优化连接 + 单艇 TSP-CPP + 3-DOF 可跟踪输出”的融合框架。

核心融合来源如下：

- `1802.03221v1.pdf`：引入面向无人艇航行安全的改进 A*，包含障碍邻域安全权重、目标引导启发项、路径节点裁剪。
- `IROS2017a.pdf`：采用 Dubins 车辆覆盖思想，将每个覆盖通道建模为带方向的访问节点，并用最小转弯半径约束计算区域间连接代价。
- `Multi-UAV_Coverage_Path_Planning_Based_on_Balanced_Graph_Partitioning.pdf`：采用区域图建模、节点覆盖时间权重、连通负载均衡图划分，用于多无人艇任务分配。
- `Efficient_Coverage_Path_Planning_and_Underwater_Topographic_Mapping_of_an_USV_Based_on_A-Improved_Bio-Inspired_Neural_Network.pdf`：吸收短路径、少转弯、漏扫补扫和 A* 跳转思想，用于局部连接与残差覆盖。
- `Joint-optimized coverage path planning framework for USV-assisted offshore bathymetric mapping From theory to practice.pdf`：吸收区域分解、候选覆盖模式、区域访问顺序和出入口联合优化，用于单艇区域间 TSP-CPP。
- `s41598-025-20978-8.pdf`：吸收凹区域分割、覆盖方向优化、TSP 与覆盖模式联合优化、3-opt/ACO 类后优化思想。

## 算法设计

1. 输入统一为任务区域、无人艇数量、初始位姿、矩形覆盖足迹、最小转弯半径、3-DOF 运动约束和优化权重。矩形覆盖足迹使用 `(lf, wf)`，条带间距固定为 `Delta = wf * (1 - rho)`，端部转弯缓冲长度为 `H >= Rmin + lf / 2 + d_safe`。

2. 区域分解采用混合策略。若任务区域为简单矩形，则直接按覆盖方向生成主区域，并按负载需要切分为子区域；若后续支持凹多边形或静态禁区，则使用扫线精确分解和基于凹点/支撑线的凸分解，并优先选择“总跨度最小、瘦长区域惩罚最小”的分解结果，以减少转弯次数。

3. 每个子区域生成多个候选覆盖模式 `RegionCoveragePattern`。候选方向来自区域长边方向、最小外接矩形方向、边界支撑线方向和初始航向近似方向。每个模式由平行覆盖条带、入口位姿、出口位姿、条带顺序、覆盖长度、转弯代价和 Dubins 可行性组成。

4. 矩形覆盖模型严格用于条带生成和覆盖验证。条带中心线间距不超过 `Delta`，条带端点向外预留 `lf / 2` 和转弯缓冲区，条带内部保持直线覆盖，条带间连接必须满足 `kappa <= 1 / Rmin`。路径规划层输出的是 3-DOF 可跟踪参考，不直接替代 NMPC。

5. 子区域覆盖时间权重定义为：

```text
W(region) = min_pattern(length / v_ref + turn_angle / r_max + dubins_turn_length / v_turn + penalty_curvature + penalty_redundancy)
```

其中 `r_max` 来自 3-DOF yaw-rate 约束，`v_turn` 根据转弯半径自动降速。

6. 多无人艇负载均衡采用带权连通图划分。节点是子区域，边是邻接关系或可通行连接，节点权重是最优候选覆盖时间。初始化使用加权 k-means 或距离种子增长，随后执行边界节点迁移和交换，保持每艘艇分配区域连通，并优化 `max(W_i)`、负载方差和跨区连接代价。

7. 改进 A* 用于区域间连接、子区域访问初始顺序和漏扫补扫跳转。A* 边代价为：

```text
edge_cost = wL * DubinsLength + wS * safety_weight + wA * heading_change + wR * Rmin_violation + wB * boundary_risk
```

启发项使用目标距离乘以目标引导因子，优先选择距离短、航向变化小、远离边界和障碍的节点。

8. 单艇区域间路径规划建模为带候选覆盖模式的单旅行商问题。每艘艇在自己的分配子图内求解 `TSP-CPP`：访问每个区域一次，同时为每个区域选择一个覆盖模式、入口和出口。初始解由改进 A* 生成，候选模式选择使用“当前出口到候选入口 Dubins 代价 + 区域内部覆盖代价 + 候选出口到下一目标预估代价”。

9. 单艇 TSP-CPP 后优化采用确定性的 2-opt/3-opt。每次交换区域顺序后，重新选择受影响区域的覆盖模式和入口/出口，若总目标函数下降则接受。大规模区域数量较多时，可增加 FA3ACO/ACO 作为可选求解器，但默认实现使用可复现实验的 A* 初始化加 2-opt/3-opt。

10. 短路径和小转弯角度统一进入目标函数：

```text
J =
  wL * total_length
+ wTheta * sum_abs_turn_angle
+ wT * total_time
+ wB * load_imbalance
+ wR * curvature_penalty
+ wC * uncovered_penalty
```

`total_length` 控制路径短，`sum_abs_turn_angle` 控制少转弯，`total_time` 控制任务效率，`load_imbalance` 控制多艇协同公平性，`curvature_penalty` 保证满足 `Rmin`，`uncovered_penalty` 保证全覆盖优先级最高。

11. 最终输出每艘艇的分层路径：区域访问序列、区域内覆盖条带、条带间 Dubins/Bezier 平滑连接、带时间戳的 3-DOF 参考轨迹。输出轨迹必须满足最小转弯半径、速度上限、yaw-rate 上限和矩形足迹覆盖完整性。

## 对外接口

新增或扩展路径规划层类型：

- `PathPlanningConfig`：包含覆盖足迹、`Rmin`、重叠率、速度约束、A* 权重、TSP 优化权重、负载均衡参数。
- `DecomposedRegion`：表示分解后的子区域，包含几何边界、邻接关系、面积、候选覆盖方向和覆盖时间权重。
- `CoveragePass`：表示一条矩形足迹覆盖条带，包含中心线、入口/出口位姿、覆盖宽度和时间估计。
- `RegionCoveragePattern`：表示一个区域的一种完整覆盖候选模式，包含条带序列、入口/出口、内部 Dubins/Bezier 连接和代价。
- `BalancedAssignment`：表示多艇区域分配结果，包含每艇区域集合、负载、连通性检查和负载差。
- `SingleUsvTourPlan`：表示单艇 TSP-CPP 结果，包含区域顺序、每区覆盖模式、连接路径和总目标值。
- `PathPlanningResult`：表示完整多艇路径规划结果，可转换为已有 `TrajectoryReference` 和仿真框架输入。

主接口保持简洁：

```python
plan_path_layer(config, mission, fleet) -> PathPlanningResult
```

内部阶段接口固定为：

```python
decompose_area(mission, config) -> list[DecomposedRegion]
build_region_graph(regions, config) -> RegionGraph
balance_workload(region_graph, fleet, config) -> BalancedAssignment
generate_region_patterns(region, config) -> list[RegionCoveragePattern]
solve_single_usv_tsp_cpp(assigned_regions, start_pose, config) -> SingleUsvTourPlan
assemble_multi_usv_references(tours, config) -> PathPlanningResult
```

## 测试计划

- 区域分解测试：矩形、长宽比极端矩形、带凹边界样例均能生成无重叠、无遗漏、邻接关系正确的子区域。
- 矩形覆盖测试：不同 `lf`、`wf`、`rho` 下，条带间距满足要求，覆盖率达到 100%，无越界覆盖或漏扫。
- 转弯半径测试：所有区域内换行连接和区域间连接满足 `kappa <= 1 / Rmin + 1e-3`，入口/出口航向连续。
- 负载均衡测试：`N=2/4/8` 时，每艇负载差默认不超过 10%，每艇分配子图保持连通。
- A* 优化测试：与普通 A* 对比，改进 A* 在相同场景下路径长度不增加明显，转弯角总和下降，安全边界距离不低于阈值。
- 单艇 TSP-CPP 测试：区域访问顺序、覆盖模式、入口/出口联合优化后，总目标函数低于最近邻初始解。
- 集成测试：将 `PathPlanningResult` 转成已有轨迹参考，接入 3-DOF/NMPC 闭环仿真，验证覆盖完成率、跟踪误差和最小安全距离。
- 漏扫补扫测试：人为删除部分覆盖单元后，残差区域能重新生成 `ResidualRegion` 或 `ResidualStripTask`，并分配给最近且最早空闲的无人艇。

## 假设与默认值

- 默认任务区仍以矩形为主；凹区域和静态禁区作为路径规划层的扩展能力预留。
- 默认无人艇同构，使用同一矩形覆盖足迹、同一 `Rmin` 和同一 3-DOF 约束。
- 高层路径规划使用 Dubins 曲线作为 3-DOF 可行性的几何代理，真实动力学跟踪由已有 NMPC 层完成。
- 动态障碍不进入离线路径规划层的区域分解，仍由已有 CBF/RVO2/NMPC 局部避障处理。
- 默认求解器采用确定性流程：区域分解、负载均衡、改进 A* 初始化、单艇 TSP-CPP、2-opt/3-opt 后优化。
- 若区域数量很大，可在后续版本启用 ACO/FA3ACO 作为可选大规模近似求解器，但不是默认路径。
