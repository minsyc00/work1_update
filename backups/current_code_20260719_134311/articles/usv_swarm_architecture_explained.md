# 多无人艇集群全覆盖路径规划与控制框架 — 架构与算法详解

> **项目名**: `usv-swarm-coverage`
> **语言**: Python ≥ 3.10
> **核心依赖**: CasADi (NMPC优化), NumPy, Matplotlib
> **目标**: 为多艘无人水面艇（USV）生成协同全覆盖路径，并在闭环仿真中执行

---

## 目录

1. [系统总览](#1-系统总览)
2. [数据模型层 (schema.py)](#2-数据模型层-schemapy)
3. [几何与运动学基础](#3-几何与运动学基础)
4. [第1层：快速全局覆盖规划 (planning.py)](#4-第1层快速全局覆盖规划-planningpy)
5. [第2层：论文融合路径规划 (path_planning/)](#5-第2层论文融合路径规划-path_planning)
6. [第3层：闭环执行与控制](#6-第3层闭环执行与控制)
7. [仿真与可视化](#7-仿真与可视化)
8. [完整算法流程总结](#8-完整算法流程总结)

---

## 1. 系统总览

整个框架由 **三层递进式架构** 组成，每一层都可以独立运行，也可以级联工作：

```
┌──────────────────────────────────────────────────────────────┐
│                     配置层 (PlannerConfig)                    │
│   任务参数 + 船队参数 + 覆盖足迹 + 优化权重 + 安全边界          │
└──────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│              第1层：快速全局覆盖规划 (planning.py)             │
│   矩形区域 → Boustrophedon条带 → DP分区 → CBS-MAPF → 平滑路径  │
│   特点：快速、无障碍物假设、适合矩形区域                         │
└──────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│         第2层：论文融合路径规划 (path_planning/)  [可选]        │
│   障碍物感知分解 → 多模式生成 → 图构建 → 负载均衡 → TSP        │
│   特点：障碍物感知、ACO/FA³ACO求解、残余回填                    │
└──────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│              第3层：闭环执行与控制                             │
│   RVO启发式 + NMPC最优控制(CasADi) + CBF约束 + 覆盖追踪        │
│   特点：实时、避碰、双模(3DOF/6DOF)估计                         │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. 数据模型层 (schema.py)

整个系统的类型基石，使用 Python `dataclass` 定义。

### 2.1 状态表示

- **`Pose2D`**：`(x, y, ψ)` — 二维位姿
- **`State3DOF`**：`(x, y, ψ, u, v, r)` — 三自由度平面状态（纵荡、横荡、艏摇）
- **`State6DOF`**：`(x, y, z, φ, θ, ψ, u, v, w, p, q, r)` — 六自由度完整状态

`State3DOF` 是主工作状态；`State6DOF` 用作"真实"动力学模型，计算与简化3DOF模型之间的不匹配度。

### 2.2 配置结构

| 结构 | 关键参数 |
|---|---|
| `MissionConfig` | 区域尺寸 `area_length_x/y`、覆盖率重叠 `overlap_ratio`、重规划/控制频率、残余覆盖开关 |
| `FleetConfig` | 初始状态、巡航/覆盖/转弯速度、最大推力/力矩、最小转弯半径、智能体数量 |
| `CoverageFootprint` | 覆盖足迹 `length_lf × width_wf`、覆盖效率 `η_cov` |
| `PlannerWeights` | NMPC代价权重：`w_pos`(位置)、`w_psi`(艏向)、`w_vel`(速度)、`w_u`(控制力)、`w_du`(控制变化)、`w_soft`(软约束) |
| `SafetyMargins` | `d_safe`(安全距离)、边界边距、`δ_safe_max`(最大不匹配补偿)、`t_block`(阻塞时间) |
| `PlannerConfig` | **顶层聚合**，包含以上全部 |

### 2.3 规划中间结构

- **`StripTask`**：一条Boustrophedon扫描线 — 起点/终点位姿、扫描轴、中心坐标、两侧穿梭口
- **`AssignmentPlan`**：每个智能体的连续条带分配区间 + 估计成本
- **`PathRequirement`**：一个路径需求（transit/cover/hold/turn），带资源ID和时间预算
- **`MAPFReservationTable`**：CBS冲突解决后的带时间窗资源预订
- **`TimedPathSegment`**：一个路径段，含起止时间和采样点
- **`TrajectoryReference`**：时间参数化的采样参考轨迹，供NMPC跟踪

### 2.4 覆盖追踪

- **`CoverageState`**：基于网格的覆盖图 — `coverage_ratio[y][x]` 累积访问次数，`covered[y][x]` 布尔标记
- **`CoverageResidual`**：未覆盖区域的连通分量

---

## 3. 几何与运动学基础

### 3.1 Dubins路径 (dubins.py)

Dubins路径是**连接两个有向位姿的最短曲率有界路径**，假设车辆只能向前行驶。本实现枚举全部6个Dubins族：

| 族 | 模式 | 含义 |
|---|---|---|
| LSL | 左转 → 直行 → 左转 | |
| RSR | 右转 → 直行 → 右转 | |
| LSR | 左转 → 直行 → 右转 | |
| RSL | 右转 → 直行 → 左转 | |
| RLR | 右转 → 左转 → 右转 | CCC类 |
| LRL | 左转 → 右转 → 左转 | CCC类 |

**算法流程** (`dubins_shortest_path`):

1. 将起点/终点坐标转换到以起点为原点的归一化坐标系（距离除以转弯半径）
2. 提取归一化参数 `(α, β, d)`：α=起始艏向与视线的夹角，β=终止艏向与视线的夹角，d=归一化距离
3. 对6个生成器分别计算归一化段长
4. 选择总长最小的路径
5. 将归一化长度乘回转弯半径，还原为实际长度

**路径采样** (`sample_dubins_path`)：沿每个段（L/R/S）以固定步长前进，调用 `advance_pose_along_mode` 累积位姿。

### 3.2 贝塞尔曲线 (geometry.py)

使用 **五次贝塞尔曲线**（6个控制点）作为轨迹平滑工具：

```
P(t) = Σ B_i^5(t) · CP_i ,  t ∈ [0,1]
其中 B_i^5 是五次伯恩斯坦基函数
```

提供：
- `bezier_point()` — 求点
- `bezier_first_derivative()` — 求切线方向（用于计算航向角）
- `bezier_second_derivative()` — 求二阶导（用于计算曲率）
- `bezier_curvature()` — κ = |P' × P''| / |P'|³
- `sample_quintic_bezier()` — 均匀采样，同时返回点、航向、最大曲率

**控制点构造**（在smoothing.py中）：

```
P0 = start
P5 = end
tangent_scale = max(turn_radius*0.85, min(segment_lengths..., 2.5*turn_radius))
P1 = P0 + tangent_scale * unit_heading(start.ψ)
P2 = P1 + 0.75 * tangent_scale * unit_heading(start.ψ)
P4 = P5 - tangent_scale * unit_heading(end.ψ)
P3 = P4 - 0.75 * tangent_scale * unit_heading(end.ψ)
```

这在起点和终点产生与Dubins路径匹配的切线方向，同时用"磁铁"效果（P2/P3向内拉）保持中点附近的光滑性。

### 3.3 几何工具 (geometry.py)

- `wrap_angle(θ)` → 将角度归一化到 `[-π, π]`
- `mean_heading(headings)` → 向量平均法计算平均方向
- `distance_xy(a, b)` → 二维欧氏距离
- `polyline_length(points)` → 折线总长
- `rotated_rectangle_mask()` → 生成旋转矩形覆盖区域的二值掩膜（用于覆盖追踪）
- `connected_components(mask)` → BFS提取二值掩膜中的所有连通区域（用于残余检测）

---

## 4. 第1层：快速全局覆盖规划 (planning.py)

这是**基础的带状覆盖流水线**，适用于无障碍物的矩形区域。入口函数是 `plan_global_coverage(config)`。

### 4.1 步骤1：生成Boustrophedon条带

```
build_boustrophedon_strips(config)
```

**Boustrophedon（牛耕式）扫描** 是全覆盖的标准方法：USV沿平行线来回扫描，相邻行的方向相反。

**算法**：

1. **选择扫描轴**：
   - 若区域长宽不等 → 沿较长边扫描
   - 若正方形 → 取智能体平均艏向的最近主轴（x或y）

2. **计算条带数**：
   ```
   effective_width = footprint_width * (1 - overlap_ratio)   // 考虑重叠
   strip_count = ceil((area_cross_width - footprint_width) / effective_width) + 1
   ```

3. **生成每条条带**：
   - 偶数索引 → 正向（沿+x 或 +y）
   - 奇数索引 → 反向（沿-x 或 -y）
   - 每条条带还附带 `pocket_left` / `pocket_right`（转弯穿梭口），供转弯/避让使用

### 4.2 步骤2：连续分区（任务分配）

```
solve_contiguous_partition(config, strips)
```

**目标**：将 `N` 条条带分成 `M` 个连续块（每个智能体一块），使**最大智能体成本最小化**（Minimax）。

**算法**：

1. 按扫描轴方向对智能体排序（使分配与初始位置对齐）
2. 计算平均覆盖时间作为基准
3. 预计算所有 `(agent, start_strip, end_strip)` 组合的成本缓存：

   ```
   block_cost = cover_time                          // 覆盖耗时
              + initial_transit_time                // 从初始位置到首条条带的Dubins过渡
              + Σ(turn_time between strips)         // 条带间的Dubins转弯
              + λ1 * turn_count                     // 转弯次数惩罚
              + λ2 * |cover_time - avg_cover_time|  // 负载不均衡惩罚
   ```

4. **动态规划** (Minimax)：

   ```
   dp[m][j] = 前m个智能体分配前j条条带的最小最大成本
   dp[m][j] = min_{i ∈ [m-1, j-1]} max( dp[m-1][i], cost(m-1, i, j-1) )
   ```

5. 通过 `choice[m][j]` 回溯，恢复每个智能体的条带区间

### 4.3 步骤3：构建路径需求

```
build_path_requirements(config, assignments)
```

将每个智能体的条带列表转化为带时间估算的需求序列：

```
transit(初始位置→首条条带) → cover(条带0) → hold(穿梭口等待) → turn(条带0→条带1) → cover(条带1) → ...
```

- **覆盖时间** = 条带长度 / 覆盖速度
- **过渡/转弯时间** = Dubins长度 / 巡航速度（或转弯速度）
- 每个需求都有一个 `resource_id`，如 `"strip:3"` 或 `"pocket:right:0"`

### 4.4 步骤4：CBS多智能体路径规划 (MAPF)

```
solve_cbs_mapf(config, requirements)
```

**基于冲突的搜索（CBS）** 是MAPF的标准最优求解器。这里的应用场景是**时间窗资源冲突**（条带和转弯穿梭口在同一时间只能被一个智能体使用）。

**算法**：

1. **根节点**：每个智能体独立调度（`_schedule_agent`），不考虑冲突
2. **优先级队列**（按makespan最小堆排序）
3. 循环：
   - 弹出一个节点
   - **寻找最早的冲突**（`_find_first_conflict`）：
     - **资源冲突**：两个智能体在同一时间使用同一资源（如同一覆盖条带）
     - **边冲突**：两个智能体相向而行（`a.from == b.to && a.to == b.from`）
   - 若无冲突 → 返回当前预订表
   - 若有冲突 → **分支成两个子节点**：
     - 子节点A：智能体A获得约束窗（禁止在该时段使用该资源）
     - 子节点B：智能体B获得约束窗
   - 每个子节点重新调度受影响的智能体，压入优先队列

4. **`_schedule_agent`**：顺次处理需求链，若遇到冲突约束，将起始时间推迟到约束结束之后

### 4.5 步骤5：构建平滑路径

```
build_smoothed_paths(config, reservations)
```

将预订表转化为几何路径：

| 需求类型 | 路径构造方式 |
|---|---|
| `cover` | 直线段（`straight_segment_points`） |
| `hold` | 驻留（重复同一位姿） |
| `turn` / `transit` | 优先尝试贝塞尔平滑 → 回退到Dubins路径 |

**贝塞尔过渡** (`_build_feasible_transition_segment`)：

1. 调用 `dubins_shortest_path` 获取参考长度
2. 以递增的切线比例（×0.8, ×1.0, ×1.25, ×1.5, ×1.8, ×2.2, ×2.8, ×3.5）尝试生成五次贝塞尔
3. 检查：
   - `max_curvature ≤ 1/turn_radius`（曲率可行性）
   - `bezier_length ≤ dubins_length × 1.15`（长度合理性）
4. 若所有贝塞尔尝试均失败 → 回退到 `sample_dubins_path`

### 4.6 步骤6：时间参数化参考轨迹

```
build_time_parameterized_references(config, paths)
```

将平滑路径转化为按时间采样的 `TrajectoryReference`，每个采样点包含：时间、位置、艏向、参考纵荡速度、参考艏摇角速度、段类型。

---

## 5. 第2层：论文融合路径规划 (path_planning/)

这是**更高级的路径规划层**，支持障碍物感知分解、多覆盖模式选择、TSP求解和残余回填。

### 5.1 入口：PathPlanningLayer

```python
layer = PathPlanningLayer()
request = layer.build_request(config, planning_result, static_obstacles, ...)
plan = layer.plan_paths(request, algorithm_name="paper_fusion_planner")
```

内部调用 `run_paper_fusion_pipeline()`，即核心流水线。

### 5.2 步骤1：障碍物场规范化

```
normalize_obstacle_field(static_obstacles, config, path_config)
```

- 以 `footprint_margin + safety_margin + inflation_extra` 膨胀每个障碍物
- 生成碰撞检测用多边形

### 5.3 步骤2：区域分解 (decomposition.py)

#### 5.3a 矩形分解（无障碍物）

```
decompose_rectangular_area(config, path_config)
```

将整个任务区域沿首选扫描轴均匀切分为条带形区域：

```
region_count = max(agent_count, strip_count)
            = 向上取整到 agent_count 的倍数
```

#### 5.3b 障碍物感知分解（有障碍物）

```
decompose_obstacle_aware_area(config, path_config, obstacle_field)
```

1. **构建切割线**：
   - 区域边界（0, Lx, 0, Ly）
   - 规则网格线（间距 = `max(resolution, footprint*2)`）
   - 每个障碍物的包围盒边界
   - 障碍物的凸出顶点坐标

2. **生成所有单元格**：对切割线的笛卡尔积
3. **过滤**：
   - 移除中心在障碍物内的单元格
   - 移除与膨胀障碍物碰撞的单元格
   - 移除小于 `min_free_cell_size` 的单元格
4. **设置扫描轴**：沿较长边的方向

### 5.4 步骤3：覆盖模式生成 (patterns.py)

```
generate_all_region_patterns(regions, config, path_config, obstacle_field)
```

对每个区域，沿每个候选扫描轴生成 **Boustrophedon覆盖模式**：

1. 计算覆盖通道数：
   ```
   pass_count = 1  (若 cross_width ≤ footprint_width)
              = ceil((cross_width - footprint_width) / effective_spacing) + 1
   ```

2. 对每个通道：
   - 若无障碍物 → 整个区间作为一个覆盖段
   - 若有障碍物 → 用 `clipped_axis_aligned_segments` 将区间**裁剪**成无碰撞子段，只保留长度 ≥ `min_pass_length` 的段

3. 偶数/奇数通道交替方向

4. **可行性检查**：
   - 通过Dubins连接相邻通道，检查碰撞和越界
   - 通过 `sampled_segment_footprint_collides` 检查覆盖段与障碍物的碰撞

5. 每个区域返回一个或多个 `RegionCoveragePattern`（按耗时排序，可行的优先）

### 5.5 步骤4：区域图构建 (graph.py)

```
build_region_graph(regions, patterns, config, obstacle_field)
```

构建 `RegionGraph`：

- **节点** = 区域，权重 = 最佳模式的估算耗时
- **边** = 相邻区域之间，权重 = Dubins过渡时间 + 艏向变化 + 碰撞惩罚
- 邻接关系由边界接触/重叠决定

### 5.6 步骤5：负载均衡分配 (assignment.py)

```
balance_region_workload(graph, config)
```

**Minimax连续分区 + 局部改进**：

1. 按空间顺序对区域排序（沿扫描轴）
2. 对智能体同样按空间顺序排序
3. **DP**：`dp[a][j] = min_{i < j} max(dp[a-1][i], cost(regions[i:j]))`
4. **边界迁移优化**（最多20轮）：
   - 找到负载最重和最轻的智能体
   - 若转移一个边界区域能减小差距且保持连通 → 执行转移
5. 返回 `BalancedAssignment`（含不均衡率 `imbalance_ratio`）

### 5.7 步骤6：多智能体TSP求解 (tsp.py + aco.py)

```
solve_multi_agent_tours(assignment, graph, config, path_config)
```

每个智能体独立求解 **TSP-CPP**（决定访问区域的顺序和每个区域的覆盖模式）。

#### 5.7a 蚁群优化 (ACO / FA³ACO)

**`solve_aco_tsp_cpp`** (aco.py)：

1. **状态表示**：蚂蚁的路径是一系列 `(region_id, pattern)` 对
2. **边代价**：
   ```
   edge_cost(prev_pattern, candidate) =
       length_weight * (dubins_transition + pattern.total_length)
     + turn_weight   * (transition_turn + pattern.turn_angle)
     + time_weight   * (transition_time + pattern.estimated_time)
     + repeat_penalty + collision_penalty + infeasible_penalty
   ```

3. **信息素**：存储在 `(from_pattern_key, to_pattern_key)` 对上
4. **选择概率**：
   ```
   P(edge) ∝ τ(edge)^α × η(edge)^β
   其中 η = 1/cost (启发式信息)
   ```

5. **FA³ACO增强**（分数阶吸引力ACO）：
   - **分数阶记忆**：将历史概率分布按分数阶导数权重衰减
     ```
     memory = Σ_{k=1}^{depth} Γ(k-ν)/(Γ(1-ν)·Γ(k+1)) · P_history[region]
     ```
   - **自适应蒸发率**：
     ```
     ρ(t) = ρ_min + (ρ_max - ρ_min) · exp(-ρ_decay · t)
     ```
   - **3-opt后优化**：对每轮最优解施加3-opt搜索

6. **精英策略**：前25%的蚂蚁 + 全局最优解沉积信息素

#### 5.7b 确定性回退求解器

若ACO失败 → 执行确定性求解：

1. **A\*初始排序**：使用 `turn_aware_astar` 贪心构建初始区域访问顺序
2. **2-opt局部搜索**：反转子序列，接受首次改进
3. **3-opt局部搜索**：尝试3种重组方式（反转B、反转C、反转B+C）

迭代次数根据区域数量自适应调整：
- 超过30个区域 → 最多1轮2-opt，0轮3-opt
- 超过18个区域 → 最多3轮2-opt，0轮3-opt

### 5.8 步骤7：段组装与连接器选择 (smoothing.py)

```
build_obstacle_aware_transition_segments(start, end, ...)
```

连接两个位姿时，按以下**层级策略**尝试：

```
第1级：直接Dubins + 贝塞尔平滑
  └─ 成功（无碰撞、曲率可行）→ 直接使用 ✓

第2级：障碍物感知A* + 走廊平滑
  ├─ 在障碍物场上运行基于网格的A*（多分辨率尝试）
  ├─ Chaikin平滑 + 曲率检查
  └─ 成功 → 使用 ✓

第3级：运动基元网格搜索
  ├─ 构建 (x, y, θ) 状态空间
  ├─ 基元：直行、左转、右转
  ├─ 检查每个基元是否与障碍物碰撞
  ├─ A*搜索（启发式 = 欧氏距离 + 转弯半径×艏向差）
  └─ 成功 → 使用 ✓

第4级：安全走廊边
  └─ 直线连接（标记 kinematic_feasible=false，后续由轨迹跟踪平滑） ⚠
```

Chaikin平滑算法：

```python
for _ in range(iterations):
    for each consecutive pair (p0, p1):
        q = 0.75*p0 + 0.25*p1   # 1/4 点
        r = 0.25*p0 + 0.75*p1   # 3/4 点
    points = [p0, q, r, p1, ...]  # 细分
```

### 5.9 步骤8：残余覆盖回填 (residual_planner.py)

```
append_residual_local_tsp(config, path_config, obstacle_field, tours)
```

1. 评估当前所有路径的覆盖状态，检测未覆盖的连通分量
2. 将每个残余组件转化为微区域 → 生成覆盖模式（含fallback单通道模式）
3. 每个智能体**贪婪地**从其当前路径末端出发：
   - 对每个残余区域计算 score = 过渡长度 + 覆盖时间 + 重复路径惩罚 + 跨智能体所有权惩罚
   - 选择得分最低的残余区域
   - 将覆盖段追加到该智能体的路径中
4. 对通过的/覆盖的区域施加**重复路径惩罚**，避免在已工作区域来回穿梭
5. 应用**跨智能体覆盖所有权惩罚**，防止智能体覆盖分配给其他智能体的区域

### 5.10 步骤9：资源时间窗调度 (scheduling.py)

```
apply_resource_window_schedule(agents, separation_time)
```

轻量级MAPF后处理：

1. 排序所有段（按起始时间）
2. 对每个段，检查其 `metadata["resource_id"]`
3. 若两个智能体的段使用相同资源且时间重叠 → 延迟后到智能体的该段及后续所有段
4. 段的时间戳通过 `retime_segment` 整体平移

### 5.11 路径规划配置 (PathPlanningConfig)

完整的可调参数集（部分重点参数）：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `tsp_solver` | `"deterministic"` | TSP求解器：deterministic / aco / fa3aco |
| `aco_ant_count` | 30 | 蚁群规模 |
| `aco_iterations` | 80 | 蚁群迭代次数 |
| `aco_alpha/beta/rho/q` | 1.0/3.0/0.35/100 | ACO经典参数 |
| `fa3aco_fractional_order` | 0.65 | 分数阶导数的阶数 |
| `fa3aco_memory_depth` | 4 | 记忆回溯深度 |
| `tsp_2opt_iterations` | 25 | 2-opt局部搜索轮数 |
| `repeat_path_penalty_weight` | 8.0 | 重复路径惩罚系数 |
| `main_repeat_path_penalty_weight` | 12.0 | 主路径重复惩罚 |
| `residual_backfill_cycles` | 3 | 残余回填最大轮数 |
| `max_residual_backfill_regions` | 12 | 每轮最多回填区域数 |

---

## 6. 第3层：闭环执行与控制

### 6.1 USV动力学模型 (USV3DOFModel)

```python
u_dot = (thrust - damp_u * u) / mass_u + mismatch[0]
v_dot = (-damp_v * v + cross_coupling * r) / mass_v + mismatch[1]
r_dot = (yaw_moment - damp_r * r) / mass_r + mismatch[2]
```

简化的3-DOF水面艇模型，包含：
- 质量矩阵（对角线）：`mass_u`, `mass_v`, `mass_r`
- 线性阻尼：`damp_u`, `damp_v`, `damp_r`
- 横向-艏摇耦合：`cross_coupling`
- 模型不匹配扰动注入：`mismatch[3]`

### 6.2 双模估计器 (Parallel6DOFEstimator)

并行运行3DOF和6DOF模型，估计两者之间的不匹配：

```
mismatch = 0.2 * (state6.u - state3.u, state6.v - state3.v, state6.r - state3.r)
δ_safe = min(δ_safe_max, 0.4*attitude_mag + 0.5*||mismatch||)
```

`δ_safe` 被叠加到标称安全距离上，在艏摇/艉倾较大或不匹配较大时提供额外裕度。

### 6.3 覆盖追踪器 (CoverageTracker)

基于网格的覆盖图：

1. 构建分辨率 = `min(footprint_width/2, footprint_length/2)` 的网格
2. `update(pose)`：在当前位置用 `rotated_rectangle_mask` 标记网格单元为已覆盖
3. `detect_residuals(min_component_cells=3)`：
   - 提取未覆盖的连通分量（BFS）
   - 仅保留 ≥ 3个单元的连通分量
   - 计算质心和包围盒

### 6.4 RVO启发式速度

在调用NMPC之前，先计算一个**偏好速度**用作NMPC代价中的软引导：

```
preferred = desired_direction * desired_speed             // 朝向参考轨迹
          + 0.6 * Σ (neighbor_repulsion)                  // 远离邻近智能体
          + 0.6 * Σ (1.5 * obstacle_repulsion)            // 远离障碍物（1.5倍权重）
```

排斥力与 `(neighbor_radius - distance) / neighbor_radius` 成正比（距离越近排斥越强）。

### 6.5 CasADi NMPC控制器 (nmpc.py)

这是整个系统最核心的控制器。它构造并求解一个**带约束的非线性最优控制问题**。

#### 6.5a 优化变量

- `x ∈ R^{6×(H+1)}`：状态轨迹 `[x, y, ψ, u, v, r]`
- `u ∈ R^{2×H}`：控制序列 `[thrust, yaw_moment]`
- `s_nei ∈ R^{N_nei×H}`：邻近智能体避碰松弛变量
- `s_obs ∈ R^{N_obs×H}`：障碍物避碰松弛变量
- `s_bound ∈ R^{4×H}`：边界约束松弛变量

#### 6.5b 代价函数

```
J = Σ_{k=0}^{H-1} [
      w_pos * ||pos_k - ref_pos_k||²              // 位置跟踪
    + w_ψ   * (1 - cos(ψ_k - ref_ψ_k))            // 艏向跟踪（余弦距离）
    + 0.5*w_vel * ||(u_k - ref_u_k, r_k - ref_r_k)||²  // 速度跟踪
    + w_vel * ||(vx_k - pref_vx_k, vy_k - pref_vy_k)||² // RVO偏好
    + w_u   * (thrust² + yaw²)                     // 控制力
    + w_du  * ||u_k - u_{k-1}||²                   // 控制平滑
    + w_soft * (s_nei² + s_obs² + s_bound²)        // 松弛变量惩罚
]
```

#### 6.5c 约束条件

**动力学**（前向欧拉离散）：
```
x_{k+1} = x_k + dt * [
    u*cos(ψ) - v*sin(ψ),
    u*sin(ψ) + v*cos(ψ),
    r,
    (thrust - damp_u*u)/mass_u + mismatch_x,
    (-damp_v*v + cross_coupling*r)/mass_v + mismatch_y,
    (yaw - damp_r*r)/mass_r + mismatch_z
]
```

**控制限幅**：
- `|thrust| ≤ max_thrust`
- `|yaw_moment| ≤ max_yaw_moment`
- `|Δthrust| ≤ 0.7*max_thrust`
- `|Δyaw| ≤ 0.7*max_yaw_moment`

**状态限幅**：
- `-0.25*cruise_speed ≤ u ≤ 1.4*cruise_speed`
- `|v| ≤ 1.4*cruise_speed`
- `|r| ≤ max(turn_speed_max/min_turn_radius, 0.2)`
- `0 ≤ x ≤ area_length_x`
- `0 ≤ y ≤ area_length_y`

**CBF风格避碰**（离散时间控制障碍函数）：
```
对每个邻近智能体 j:
    h_k     = ||pos_k - neigh_j,k||² - safe_distance²      ≥ -s_nei[j,k]
    h_{k+1} - (1-γ)*h_k + s_nei[j,k]                       ≥ 0
其中 γ = 0.4 控制衰减速度

对每个障碍物 j:
    h_k     = ||pos_k - obs_j,k||² - (safe_distance + obs_radius)²  ≥ -s_obs[j,k]
    h_{k+1} - (1-γ)*h_k + s_obs[j,k]                                  ≥ 0

对4个边界方向:
    h_k     = dist_to_boundary_k - boundary_margin          ≥ -s_bound[i,k]
    h_{k+1} - (1-0.5)*h_k + s_bound[i,k]                   ≥ 0
```

这保证了若 `h_k ≥ 0`，则 `h_{k+1} ≥ (1-γ)h_k`，即安全距离呈指数衰减但不立即违反。

#### 6.5d 求解器选择

- 首选：**IPOPT**（内点法），最多60次迭代，容差1e-3
- 回退：**SQP**（序列二次规划）

#### 6.5e 热启动

- 将上一解的轨迹向前平移一步，末尾重复最后一帧
- 初始猜测的状态来自参考轨迹，控制来自参考速度

### 6.6 SwarmRuntime 控制循环

```
control_step(agent_state, shared_predictions, obstacle_tracks):
  1. 提取当前智能体的参考窗口（horizon_steps帧）
  2. 归一化其他智能体的预测轨迹（实际预测或回退到参考轨迹）
  3. 估计模型不匹配和 δ_safe
  4. 计算RVO偏好速度
  5. 调用 CasADi NMPC 求解
  6. 若 NMPC 失败 → 进入 safe_hold 模式（计算避险方向+PD控制）
  7. 更新覆盖图
  8. 检测残余
  9. 返回控制指令 + 安全状态 + 预测轨迹
```

`safe_hold` 降级策略：
```
threat = Σ (offset/distance) for all neighbors + 1.5*Σ (offset/distance) for obstacles
如果 threat ≈ 0 → 指向区域中心
steering = clip(2.5*(desired_heading - ψ) - 0.8*r, ±max_yaw_moment)
thrust   = clip(-1.2*u, ±max_thrust)
```

---

## 7. 仿真与可视化

### 7.1 闭环仿真 (simulation.py)

```
simulate_swarm_closed_loop(config, planning_result, obstacle_tracks, total_time)
```

**主循环**：

```python
for step in range(steps):
    # 每个智能体独立计算控制指令
    for agent_id:
        result = runtime.control_step(runtime_state, shared_predictions, obstacle_tracks)
        # 共享预测轨迹供其他智能体使用

    # 推进所有智能体状态
    for agent_id:
        next_state = model.step(current_state, result.cmd, dt, mismatch)
        next_state6 = propagate_state6_surrogate(state6, next_state3)
        coverage_tracker.update(next_state.pose())

    # 检测残余
    coverage_tracker.detect_residuals()

    # 早停条件：覆盖率 ≥ 99.5% 且所有智能体已完成参考轨迹
    if coverage >= 0.995 and all agents done:
        break
```

### 7.2 动画渲染

`render_simulation_animation()` 生成GIF动画，包含：
- 覆盖图（蓝色热力图）
- 智能体轨迹（彩色折线）
- 足迹多边形（旋转矩形）
- 参考轨迹（彩色虚线）
- 条带线（灰色点线）
- 动态障碍物（红色圆圈）
- 实时统计（时间、覆盖率、警告数）

### 7.3 动态障碍物

`build_crossing_obstacle_scenario()` 生成两个穿越任务区域的线性移动障碍物，用于测试避碰能力。

`_sample_obstacle()` 按时间线性插值获取障碍物位置（支持超出采样范围的外推）。

---

## 8. 完整算法流程总结

### 端到端数据流

```
PlannerConfig
    │
    ├──→ [planning.py] plan_global_coverage()
    │       ├── build_boustrophedon_strips()        生成Boustrophedon扫描线
    │       ├── solve_contiguous_partition()         DP Minimax分配
    │       ├── build_path_requirements()            构建需求序列
    │       ├── solve_cbs_mapf()                     CBS冲突解决
    │       ├── build_smoothed_paths()               贝塞尔/Dubins平滑
    │       └── build_time_parameterized_references() 时间采样
    │            │
    │            ▼
    │       PlanningResult
    │            │
    │            ├──→ [control.py] SwarmRuntime
    │            │       ├── CoverageTracker        覆盖追踪
    │            │       ├── Parallel6DOFEstimator  模型不匹配估计
    │            │       ├── RVO heuristic          偏好速度
    │            │       ├── CasADi NMPC            最优控制
    │            │       └── Safe hold fallback     安全降级
    │            │
    │            └──→ [simulation.py] simulate_swarm_closed_loop()
    │                    └── render_simulation_animation() → GIF
    │
    └──→ [path_planning/] PathPlanningLayer (可选第2层)
            ├── normalize_obstacle_field()           障碍物膨胀
            ├── decompose_*_area()                   区域分解
            ├── generate_all_region_patterns()       多模式生成
            ├── build_region_graph()                 图构建
            ├── balance_region_workload()           负载均衡
            ├── solve_multi_agent_tours()            ACO/FA³ACO + 2-opt/3-opt
            │       └── build_obstacle_aware_transition_segments()
            │               ├── 直接贝塞尔 → A*走廊 → 运动基元 → 安全边
            ├── append_residual_local_tsp()          残余回填
            └── apply_resource_window_schedule()    资源时间窗调度
                     │
                     ▼
                MultiAgentPathPlan
```

### 关键算法的计算复杂度

| 算法 | 复杂度 | 备注 |
|---|---|---|
| Dubins最短路径 | O(1) | 6族 × 常数计算 |
| Boustrophedon条带生成 | O(N_strips) | |
| DP连续分区 | O(M²·N) | M=智能体数, N=条带数 |
| CBS-MAPF | 指数级最坏情况 | 实践中冲突数远小于理论值 |
| 贝塞尔平滑 | O(N_scales × N_samples) | 最多8个缩放尝试 |
| 障碍物感知分解 | O(C² · V) | C=切割线数, V=障碍物顶点数 |
| 覆盖模式生成 | O(R · P² · S) | R=区域数, P=通道数, S=Dubins采样 |
| 负载均衡DP | O(K²·R) | K=智能体数, R=区域数 |
| ACO TSP | O(I · A · R²) | I=迭代, A=蚂蚁数, R=区域数 |
| 2-opt TSP | O(I · R²) | I=迭代 |
| A* 障碍物避开 | O(G²·logG) | G=网格节点数 |
| 运动基元网格搜索 | O(S³·B) | S=状态数, B=基元数 |
| CasADi NMPC (IPOPT) | O(H·(nx+nu)³) | H=时域长度, 实际取决于IPOPT |
| 覆盖追踪更新 | O(G) | G=网格单元数 |
| 残余检测 (BFS) | O(G) | |
| 残余回填 | O(N_r · A · P) | N_r=残余区域, A=智能体, P=模式 |
