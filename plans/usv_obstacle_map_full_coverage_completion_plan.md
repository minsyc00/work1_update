# 障碍地图全覆盖增强实现计划

## Summary

目标是在当前 `50 x 50` 静态障碍地图上，将路径规划算法、无人艇运动约束、Dubins/Bezier 曲线平滑、MAPF/CBS 调度和残差补扫闭环串起来，使多无人艇能够对障碍自由空间完成全覆盖任务。

## Step 1: 地图 JSON 读入 Planner

新增地图加载模块，读取：

```text
maps\static_obstacle_map_50x50_simple\static_obstacle_map_50x50_simple.json
```

实现内容：

- 解析 `mission_area` 为 `MissionConfig`。
- 解析 `coverage_footprint` 为 `CoverageFootprint`。
- 解析 `motion_constraints.min_turn_radius` 写入 `FleetConfig`。
- 将 `rectangle / ellipse / polygon` 转为 `StaticObstacle[]`。
- 无人艇数量、初始位姿、速度、推力等由实验配置单独提供，不写入地图 JSON。

验收：

- 可以通过 `load_map_for_planner(map_json, fleet_config)` 得到 `PlannerConfig + StaticObstacle[]`。
- 地图 JSON 不包含无人艇信息，但 planner 能组合地图和 fleet 参数运行。

## Step 2: 升级自由空间分解

目标是减少当前 axis-aligned cell 分解过度保守导致的可覆盖区域丢失。

实现内容：

- 基于障碍膨胀 polygon 的顶点、边界和任务区域边界生成扫线分割线。
- cell 判定从“整块矩形不碰障碍”升级为“cell 内部采样估计自由空间占比”。
- 支持障碍边界附近的小 cell 保留或合并，避免碎片过多。
- 对每个 cell 记录真实自由空间面积估计、邻接边和窄通道宽度。

验收：

- 自由空间面积估计明显接近 `mission_area - inflated_obstacle_area`。
- 障碍周围不再大面积丢失可覆盖空间。
- cell 邻接图保持连通或明确输出不可达组件诊断。

## Step 3: 精确障碍裁剪覆盖条带

目标是让每个 cell 内的覆盖条带只覆盖自由空间，并尽量达到自由空间 100% 覆盖。

实现内容：

- 对每条候选覆盖线与障碍 polygon 求交，得到多个自由区间。
- 每个自由区间生成一个 `CoveragePass`。
- 矩形覆盖足迹扫掠时检查是否进入膨胀障碍。
- 对障碍边缘附近的残余小区域生成短补扫 pass。
- 覆盖率评价只统计自由空间，不统计障碍内部。

验收：

- 静态障碍地图覆盖率目标提升到接近 `1.0`。
- 不存在覆盖中心线穿越障碍。
- 不存在矩形足迹进入膨胀障碍。

## Step 4: Dubins 碰撞失败时调用 A*

目标是区域间连接不再只依赖 Dubins 直连。

实现内容：

- 先尝试 Dubins 连接。
- 若 Dubins 采样点或矩形足迹碰撞，则调用 obstacle-aware grid A*。
- A* 代价包含路径长度、障碍安全权重、航向变化、边界风险和窄通道惩罚。
- A* 输出 waypoint corridor。

验收：

- 障碍阻挡两区域直连时，可以自动生成绕障连接。
- 连接路径不穿越膨胀障碍。
- A* 结果记录到 segment metadata 中。

## Step 5: A* 通道转 Dubins/Bezier 可跟踪子段

目标是 A* 结果不只是折线，而是变成 3-DOF 可跟踪路径。

实现内容：

- 对 A* waypoint corridor 做路径简化。
- 将折线拐点转换为带航向的 `Pose2D` 序列。
- 相邻 pose 之间生成 Dubins 子段。
- 每个 Dubins 子段尝试 Bezier 平滑。
- 若 Bezier 曲率不可行，则回退 Dubins。
- 所有子段输出 `PathSegmentSpec` 和 `PathWaypoint`。

验收：

- 所有子段满足 `kappa <= 1 / Rmin + 1e-3`。
- 输出路径连续、航向连续、时间单调。
- A* 绕障连接可以被转换为 `TrajectoryReference`。

## Step 6: 新路径重新接入 MAPF/CBS

目标是多艇在障碍密集地图中避免时空资源冲突。

实现内容：

- 从新 `AgentPathPlan` 生成 `PathRequirement`。
- 给窄通道、区域入口、转弯口袋、覆盖条带和绕障 corridor 分配 `resource_id`。
- 复用或扩展现有 CBS，处理资源冲突和反向边冲突。
- CBS 输出等待段、调度后时间窗和无冲突路径。
- 将调度结果重新写回 `PathWaypoint.time`。

验收：

- 多艇不会同时占用同一窄通道。
- 多艇不会在同一入口/转弯口袋发生时窗冲突。
- 调度后路径仍可转换为轨迹参考。

## Step 7: 残差补扫闭环

目标是从“检测和分配残差”升级为“自动二次规划补扫”。

实现内容：

- 根据执行轨迹或规划轨迹更新 `CoverageState`。
- 检测自由空间未覆盖残差。
- 将残差聚类成 `ResidualRegion`。
- 对残差区域重新生成覆盖 pass。
- 分配给最近且最早空闲无人艇。
- 对补扫任务执行局部 TSP-CPP 和 MAPF/CBS 调度。
- 将补扫路径追加到对应无人艇轨迹后。

验收：

- 人为制造漏扫块后能自动生成补扫路径。
- 补扫后自由空间覆盖率达到或接近 `100%`。
- 补扫路径不穿越障碍，满足 3-DOF 和转弯半径约束。

## Test Plan

- 地图加载测试：JSON 能构造 `PlannerConfig + StaticObstacle[]`。
- 分解测试：复杂障碍附近自由空间不被过度丢弃。
- 覆盖测试：自由空间覆盖率接近 `1.0`。
- 绕障连接测试：Dubins 碰撞时自动调用 A*。
- 子段转换测试：A* corridor 可转为 Dubins/Bezier 轨迹。
- MAPF 测试：窄通道和入口无时窗冲突。
- 残差测试：漏扫区域可自动补扫。
- 集成测试：`static_obstacle_map_50x50_simple` + `USV=3` + `lf=4,wf=2,Rmin=2` 完成全覆盖规划。

## Assumptions

- 地图 JSON 只保存静态地图资产，不保存无人艇初始位姿。
- 无人艇数量、初始位姿、速度、控制约束由实验配置提供。
- 当前阶段只处理静态障碍。
- 动态障碍后续再进入时空规划和局部避障联动。
