# 5.7 多智能体TSP求解 — 深度详解

> 本节对应 [tsp.py](src/usv_swarm/path_planning/tsp.py) 和 [aco.py](src/usv_swarm/path_planning/aco.py)

---

## 目录

1. [问题建模](#1-问题建模)
2. [总体架构](#2-总体架构)
3. [边代价函数](#3-边代价函数)
4. [ACO/FA³ACO 蚁群求解器](#4-acofa³aco-蚁群求解器)
5. [确定性回退求解器](#5-确定性回退求解器)
6. [段组装（Segment Assembly）](#6-段组装segment-assembly)

---

## 1. 问题建模

### 1.1 什么是 TSP-CPP？

在传统 TSP 中，旅行商需要找到访问所有城市并返回起点的最短回路。这里的**变种（TSP-CPP，即覆盖路径规划旅行商问题）**更复杂：

- 每个"城市"是一个**区域（Region）**，必须被 USV 的覆盖足迹**完全扫描**
- 每个区域有多种**覆盖模式（Coverage Pattern）**可选——不同的扫描方向（x轴/y轴）、不同的入口/出口位姿
- 边代价不是固定的——从模式A的出口到模式B的入口需要一条**带曲率约束的过渡路径**（Dubins + 避障）

因此，TSP求解器需要**同时决定**：
1. **访问顺序**（区域的排列）
2. **模式选择**（每个区域选哪个覆盖模式）
3. **过渡路径**（模式间如何连接）

### 1.2 搜索空间

假设有 `R` 个区域，每个区域有 `P` 种模式（通常 P = 1~2），则求解空间为：

```
O(R! × P^R)  种可能的解
```

即使对小规模问题（R=8, P=2），也是 ~40,320 × 256 ≈ 10³² 量级。因此需要**启发式/元启发式**方法求解。

---

## 2. 总体架构

```python
solve_multi_agent_tours(assignment, graph, config, path_config) → Dict[int, SingleUsvTourPlan]
```

每个智能体**独立**调用 `solve_single_usv_tsp_cpp()`：

```
solve_single_usv_tsp_cpp(agent_id, region_ids, graph, config, path_config)
│
├── 第一步：尝试 ACO / FA³ACO 求解（若 tsp_solver ≠ "deterministic"）
│   ├── 成功 → 直接返回 ACO 找到的巡游方案
│   └── 失败 → 记录失败原因，进入回退流程
│
└── 第二步：确定性回退求解
    ├── 2a. A* 种子初始排序 (astar_seeded_order)
    ├── 2b. 2-opt 局部搜索
    ├── 2c. 3-opt 局部搜索
    └── 2d. 段组装 (assemble_segments)
```

**设计原则**：ACO 先行，因为它在较大搜索空间中表现更好；确定性求解器作为可靠的回退方案，保证任何情况下都有可行解。

---

## 3. 边代价函数

边代价是 TSP 求解器的**核心**，它定义了从一个模式（或起始位姿）过渡到另一个模式的成本。

```python
def edge_cost(previous: RegionCoveragePattern | None, candidate: RegionCoveragePattern) -> float:
```

### 3.1 代价组成

```
edge_cost = 
    length_weight × (dubins_transition_length + pattern.total_length)     // 路径长度
  + turn_weight   × (transition_turn_angle  + pattern.turn_angle)         // 转弯角度
  + time_weight   × (transition_time        + pattern.estimated_time)     // 时间估算
  + repeat_penalty                                                         // 模式内重复路径惩罚
  + collision_penalty                                                      // 过渡碰撞惩罚 (1e6)
  + infeasible_penalty                                                     // 不可行模式惩罚 (1e6)
```

其中默认权重为：
```python
length_weight = 1.0
turn_angle_weight = 0.35
time_weight = 1.0
```

### 3.2 过渡时间计算

从 `previous.exit_pose` 到 `candidate.entry_pose`：
```python
transition = dubins_shortest_path(previous.exit_pose, candidate.entry_pose, min_turn_radius)
transition_time = transition.total_length / cruise_speed
transition_turn  = Σ (segment_length / turn_radius)  for L/R modes
```

### 3.3 碰撞惩罚

如果图中有障碍物字段，检查从出口到位姿到入口位姿的线段是否碰撞：
```python
if polyline_collides_with_obstacles([exit, entry], obstacle_field, inflated=True):
    collision_penalty = 1e6   # 硬惩罚，确保ACO不选此路径
```

### 3.4 成本缓存

所有边代价在计算后被缓存到 `edge_cache` 字典中，key 为 `(from_pattern_key, to_pattern_key)`：

```python
def cached_cost(previous, candidate):
    key = (node_key(previous), node_key(candidate))  # e.g. ("__start__", "region_0:pattern_x")
    if key not in edge_cache:
        edge_cache[key] = edge_cost_fn(previous, candidate)
    return edge_cache[key]
```

---

## 4. ACO/FA³ACO 蚁群求解器

**文件**：[aco.py](src/usv_swarm/path_planning/aco.py)

### 4.1 算法入口

```python
def solve_aco_tsp_cpp(region_ids, patterns, start_pose, path_config, edge_cost_fn, solver) → AcoTspResult
```

支持的求解器：
- `"aco"`：标准蚁群优化
- `"fa3aco"`：分数阶吸引力蚁群优化（Fractional Attractive ACO）

### 4.2 解的结构

```python
@dataclass
class _AcoSolution:
    route: List[RegionCoveragePattern]  # 按顺序排列的（区域, 模式）对
    objective: float                    # 总代价
    edge_keys: List[Tuple[str, str]]   # 信息素沉积的边

    @property
    def region_order(self) -> List[str]:       # 纯区域ID顺序
    @property
    def selected_patterns(self) -> Dict[str, RegionCoveragePattern]:  # 每个区域的选中模式
```

一只蚂蚁的路径就是完整的一条 `[pattern₀, pattern₁, ..., pattern_{R-1}]` 序列，包含了顺序和模式选择两个信息。

**关键**：搜索空间不是 "区域 → 区域"，而是 "模式 → 模式"。这意味着同一区域的两个不同扫描方向模式被视为图中的不同节点，它们共享同一个 `region_id`。蚂蚁每次选择一个模式后，该模式对应的区域即被标记为"已访问"。

### 4.3 初始化解：贪心构造

```python
def _greedy_solution(region_ids, patterns, cost_fn) → _AcoSolution | None
```

从起始位姿出发，每一步都选择 `cost(current, candidate)` 最小的候选（模式, 区域），将该区域从待访问集合中移除。

这是 ACO 的起始解，也用于计算 `initial_objective` 以衡量 ACO 的改进幅度。

### 4.4 蚂蚁构造（Ant Construction）

```python
def _construct_ant_solution(region_ids, patterns, cost_fn, pheromone, path_config, rng, use_fractional_memory)
```

这是 ACO 的核心——每只蚂蚁如何构建一条完整路径：

```
蚂蚁从起始位姿出发（current = None）
while 还有区域未访问:
    1. 枚举所有（剩余区域 × 该区域的所有模式）的组合
    2. 对每个（区域, 模式）候选：
        cost = cost_fn(current, candidate)
        η = 1 / max(cost, 1e-9)                          // 启发式信息
        τ = pheromone.get(edge_key, 1.0)                  // 信息素浓度
        desirability = τ^α × η^β                         // 选择吸引力
    3. 若 use_fractional_memory：
        memory = fractional_memory(candidate, history)      // 分数阶记忆增强
        desirability *= (1.0 + memory)
    4. 归一化 desirability 为概率分布
    5. 按概率加权随机选择一个（区域, 模式）
    6. 将该区域从 remaining 中移除
    7. current = selected_pattern
```

**选择概率**（轮盘赌）：
```
P(edge_k) = τ_k^α × η_k^β / Σ_j τ_j^α × η_j^β
```

其中默认参数：
```python
aco_alpha = 1.0    # 信息素重要性
aco_beta  = 3.0    # 启发式信息重要性（较大值意味着更依赖贪心选择）
```

### 4.5 信息素更新

```python
def _update_pheromone(pheromone, iteration_solutions, best_solution, rho, q)
```

**蒸发**（所有边）：
```
τ_{ij} ← max( (1 - ρ) × τ_{ij}, 1e-9 )
```

**沉积**（精英蚂蚁 + 全局最优）：
```python
elite_count = max(1, len(solutions) // 4)  # 前25%
for solution in top_elite:
    for edge in solution.edge_keys:
        τ[edge] += q / solution.objective

# 额外奖励全局最优
for edge in best.edge_keys:
    τ[edge] += 0.5 × q / best.objective
```

### 4.6 FA³ACO 增强

FA³ACO（**Fractional Attractive ACO**）在标准 ACO 基础上增加了三个机制：

#### 4.6a 分数阶记忆（Fractional Memory）

并非仅依赖信息素，FA³ACO 还维护一个**衰减概率记忆**，基于**分数阶微积分**的概念：

```python
def _fractional_memory(candidate_key, probability_history, path_config):
    depth = fa3aco_memory_depth  # 默认 4
    ν = fa3aco_fractional_order   # 默认 0.65

    memory = 0.0
    for k in 1..depth:           # k 是回溯步数
        weight = |Γ(k-ν) / (Γ(1-ν) × Γ(k+1))|
        memory += weight × P_history_{t-k}[candidate]
    return memory
```

**直观理解**：这是一种加权移动平均，权重服从分数阶衰减。ν=0 时等同于普通指数衰减；ν→1 时给予更久远的历史更高权重（长记忆）。0.65 是平衡短期和长期记忆的经验值。

这个记忆被用来**调节选择概率**：
```
adjusted_desirability = desirability × (1.0 + memory)
```

#### 4.6b 自适应蒸发率

标准 ACO 使用固定的蒸发率 ρ。FA³ACO 采用**指数衰减的蒸发率**：

```python
ρ(t) = ρ_min + (ρ_max - ρ_min) × exp(-ρ_decay × t)
```

其中默认值：
```python
fa3aco_rho_min   = 0.08
fa3aco_rho_max   = 0.55
fa3aco_rho_decay = 0.035
```

**直观理解**：早期迭代时蒸发率高（~0.55），信息素快速更新，鼓励探索；后期蒸发率降低（→0.08），信息素趋于稳定，利于收敛到最优解。

#### 4.6c 3-opt 后优化

每轮迭代的最优解可额外施加一次 3-opt 搜索：

```python
if fa3aco_enable_3opt:
    improved = _three_opt_improve(iteration_best, patterns, cost_fn, max_candidates=80)
    if improved.objective < iteration_best.objective:
        iteration_best = improved
        accepted_3opt += 1
```

3-opt 尝试三种重组方式（对子序列 i..j..k 的三段 A、B、C、D）：
```python
candidates = [
    A + reversed(B) + C + D,           # 反转B
    A + B + reversed(C) + D,           # 反转C
    A + reversed(C) + reversed(B) + D,  # 反转B和C
]
```

### 4.7 ACO 成功 / 失败判定

ACO 成功返回的条件：
1. 贪心初始化成功（存在有限代价的路径）
2. ACO 构造的路径的段组装（segment assembly）成功——即区域间的过渡路径经障碍物检查后均为可行的

若 ACO 失败（任何区域缺少候选模式、贪心无解、或段组装不完整），则进入确定性回退流程。

---

## 5. 确定性回退求解器

当 ACO 失败或 `tsp_solver = "deterministic"` 时使用。

### 5.1 A* 种子初始排序

```python
def _astar_seeded_order(start_pose, region_ids, graph, config, path_config) → List[str]
```

目标是生成一个**合理的区域访问顺序**，作为 2-opt/3-opt 的初始解。

**算法流程**：

```
current_pose = start_pose (智能体初始位姿)
current_region = None (还未在任何区域)
remaining = set(region_ids)

while remaining:
    if current_region is None:
        # 首次选择：找距离起始位姿最近（按最佳入口过渡成本）的区域
        best = argmin { best_entry_transition_cost(current_pose, patterns[region]) }
        astar_path = [best]
    else:
        # 后续选择：对每个剩余区域，在区域图上跑 turn_aware_astar
        candidates = []
        for region in remaining:
            result = turn_aware_astar(graph, current_region, region, ...)
            candidates.append( (result.cost, result.path) )
        best_cost, best_path = min(candidates)

    # 沿 A* 路径访问区域
    for region in astar_path:
        if region not in remaining: continue
        order.append(region)
        # 选择该区域的最佳模式（最小化当前位姿到入口的过渡成本）
        best_pattern = argmin {
            transition_length(current_pose, pattern.entry)
            + pattern.estimated_time
            + repeat_penalty
        }
        current_pose = best_pattern.exit_pose
        current_region = region
        remaining.remove(region)
```

**关键**：这里使用了 `turn_aware_astar()`——一种考虑转弯成本的 A* 变体（见后文 §5.4），它在区域图上搜索从一个区域到另一个区域的最优路径，而不仅是欧氏距离最近。

### 5.2 2-opt 局部搜索

```python
for _ in range(bounded_2opt_iterations):   # 默认25次，大区域数时自适应缩减
    for i in range(n-2):
        for j in range(i+2, n+1):
            candidate = order[:i] + reversed(order[i:j]) + order[j:]
            # 评估候选顺序
            selected, length, turn, time, obj = evaluate_order(candidate, ...)
            if obj < best_obj:
                best_order = candidate    # 接受首次改进
                improved = True
                break
    if not changed: break                 # 无改进则退出
```

**自适应迭代次数**：
```python
if region_count > 30:  → 最多1轮
if region_count > 18:  → 最多3轮
else:                  → 按配置值（默认25）
```

### 5.3 3-opt 局部搜索

```python
for _ in range(bounded_3opt_iterations):
    for i in range(1, n-2):
        for j in range(i+1, n-1):
            for k in range(j+1, n):
                for candidate in [A+rev(B)+C+D, A+B+rev(C)+D, A+rev(C)+rev(B)+D]:
                    if objective(candidate) < best_obj: accept
```

区域数超过12时不执行3-opt（O(n³) 代价过高）。

### 5.4 转弯感知 A* (turn_aware_astar)

**文件**：[astar.py](src/usv_swarm/path_planning/astar.py)

在区域图上运行 A*，但边代价不仅仅是距离——还惩罚**艏向变化**和**边界风险**：

```python
cost = base_edge_weight × safety_weight            // 基础过渡成本 × 安全放大
     + astar_heading_weight × heading_change       // 转弯惩罚 (0.35)
     + astar_safety_weight × (safety_weight - 1.0) // 安全成本 (0.5)
     + astar_boundary_weight × boundary_risk       // 边界风险 (0.2)
     + curvature_weight × curvature_violation       // 曲率违规 (100.0)
```

**安全权重**（巡航安全加权）：
```python
def sailing_safety_weight(danger_neighbor_count):
    if danger <= 0: return 1.0
    return 1.0 + 0.5 × 2^(danger-1)   # 指数增长：1个危险邻居→1.5, 2个→2.0, 3个→3.0
```

**启发式引导因子**（目标领航因子）：
```python
def goal_pilot_factor(theta):
    return 3.0 / max(4.0 - sin(theta), 1e-6)
```
其中 `theta` 是节点到目标的视线角。这一项在特定角度下放大启发式，使搜索更能体现方向敏感性。

**标准 A* 框架**：
```python
open_set = [(priority, serial, node_id)]    # 最小堆
while open_set:
    current = pop(open_set)
    if current == goal: return reconstruct_path()
    for neighbor in adjacency[current]:
        tentative_g = g[current] + edge_cost(current, neighbor, heading, ...)
        if tentative_g < g[neighbor]:
            g[neighbor] = tentative_g
            heading_to[neighbor] = new_heading
            priority = tentative_g + euclidean_heuristic * goal_pilot_factor
            push(open_set, (priority, serial++, neighbor))
```

### 5.5 目标函数评估

```python
def evaluate_order(order, start_pose, graph, config, path_config):
    current_pose = start_pose
    total_length = total_turn = total_time = 0.0

    for region_id in order:
        # 前瞻选择模式（考虑当前位姿和下一个区域）
        pattern = select_pattern(region_id, current_pose, next_region, ...)

        # Dubins过渡 + 模式内覆盖
        transition = dubins_shortest_path(current_pose, pattern.entry_pose, ...)
        total_length += transition.length + pattern.total_length
        total_turn   += transition_turn_angle + pattern.turn_angle
        total_time   += transition.time + pattern.estimated_time
        current_pose = pattern.exit_pose

    objective = length_weight × total_length + turn_weight × total_turn + time_weight × total_time
    return selected_patterns, total_length, total_turn, total_time, objective
```

**模式选择**（前瞻）：
```python
def select_pattern(region_id, current_pose, next_region, ...):
    best = argmin {
        pattern.estimated_time                                          // 内部覆盖时间
      + dubins_length(current_pose, pattern.entry_pose) / cruise_speed  // 到达成本
      + collision_penalty(current_pose → entry)                         // 碰撞惩罚
      + lookahead_cost(pattern, next_region)                            // 离开成本
      + repeat_penalty                                                  // 重复惩罚
      + (0 if feasible else 1e6)                                        // 不可行惩罚
    }
```

其中 `lookahead_cost` 是从当前模式的出口到下一个区域任一候选模式入口的最小 Dubins 过渡时间——这使模式选择具有**前瞻性**，而不仅是贪心看当前。

---

## 6. 段组装（Segment Assembly）

得到区域顺序和模式选择后，需要**实际生成几何路径**。

### 6.1 逐段贪心组装

```python
def _assemble_segments(agent_id, order, selected, start_pose, graph, config, path_config) → List[PathSegmentSpec]

segments = []
current_pose = start_pose
current_time = 0.0
remaining = list(order)

while remaining:
    # 对每个剩余区域，尝试生成从当前位置到该区域的完整段序列
    feasible_candidates = []
    for idx, region_id in enumerate(remaining):
        pattern = selected[region_id]
        region_segments = _build_region_segments_atomic(
            agent_id, region_id, pattern, current_pose, current_time, ...
        )
        if region_segments is None:     # 过渡路径不可行
            continue

        # 计算与已规划路径的重复重叠
        repeat_score = score_repeat_overlap(
            [s for s in region_segments if s.kind != "cover"],  # 只对非覆盖段检查重复
            segments,
            penalty_weight = main_repeat_path_penalty_weight    # 默认12.0
        )
        feasible_candidates.append( (repeat_score.penalty, idx, region_id, pattern, region_segments) )

    # 选择重复惩罚最小的候选
    _, accepted_index, accepted_region, accepted_pattern, accepted_segments = min(feasible_candidates)

    segments.extend(accepted_segments)
    remaining.pop(accepted_index)

return segments
```

### 6.2 原子区域段构造

```python
def _build_region_segments_atomic(agent_id, region_id, pattern, current_pose, current_time, ...):
    segments = []

    # 1. 过渡段：当前位置 → 区域入口
    transit_segments = build_obstacle_aware_transition_segments(
        start=current_pose, end=pattern.entry_pose, kind="transit", ...
    )
    if not motion_feasible(transit_segments): return None   # 运动学不可行 → 跳过
    segments.extend(transit_segments)

    # 2. 覆盖段 + 内部转弯
    for pass_idx, coverage_pass in enumerate(pattern.passes):
        cover = build_cover_segment(start=pass.start, end=pass.end, ...)
        segments.append(cover)

        if pass_idx < len(pattern.passes)-1:
            next_pass = pattern.passes[pass_idx+1]
            turns = build_obstacle_aware_transition_segments(
                start=pass.end, end=next_pass.start, kind="turn", ...
            )
            if not motion_feasible(turns): return None
            segments.extend(turns)

    return segments
```

### 6.3 重复路径惩罚

```python
def score_repeat_overlap(candidate_segments, existing_segments, penalty_weight):
    grid = shared_resource_grid_size  # 默认1.0m
    occupied = set()  # 已占用网格单元

    # 收集所有已有段的网格单元
    for segment in existing_segments:
        for point in sample_segment_points(segment, grid):
            occupied.add(quantized_point(point, grid))

    # 检查候选段与已占用单元的重叠
    overlap = 0.0
    for segment in candidate_segments:
        hits = count(point in occupied for point in sample_points(segment, grid))
        ratio = hits / len(points)
        overlap += segment.length × ratio

    return RepeatOverlapScore(
        overlap_length = overlap,
        penalty = penalty_weight × overlap    # 默认12.0 × 重叠长度
    )
```

这确保了 TSP 段组装不会让智能体在已有的覆盖/过渡路径上重复穿行。

---

## 总结：数据流图

```
region_ids, graph, start_pose, config, path_config
│
├───────────────────────────────────────────────────
│ solve_single_usv_tsp_cpp()
│
├─ [ACO路径] ──────────────────────────────────────
│  solve_aco_tsp_cpp()
│  ├── greedy_solution()          贪心初始解
│  ├── for iteration 1..80:
│  │   for ant 1..30:
│  │     construct_ant_solution()  轮盘赌选择模式
│  │       └── P ∝ τ^α × η^β
│  │       └── [FA³ACO] + 分数阶记忆
│  │   [FA³ACO] 3-opt改善本轮最优
│  │   update_pheromone()          蒸发+精英沉积
│  │       └── [FA³ACO] 自适应蒸发率 ρ(t)
│  ├── 返回: region_order + selected_patterns
│  └── _tour_from_selected_order()
│       └── _assemble_segments()   生成几何路径
│
└─ [确定性回退] ──────────────────────────────────
   ├── _astar_seeded_order()
   │    └── turn_aware_astar()    转弯感知A*
   │         └── cost = base×safety + heading + risk
   ├── 2-opt 局部搜索             反转子序列
   ├── 3-opt 局部搜索             3段重组
   └── _assemble_segments()
        └── for each region:      贪心+重复惩罚
             ├── build_obstacle_aware_transition()
             └── build_cover_segment()
                     │
                     ▼
              SingleUsvTourPlan
              (region_order + selected_patterns + segments)
```
