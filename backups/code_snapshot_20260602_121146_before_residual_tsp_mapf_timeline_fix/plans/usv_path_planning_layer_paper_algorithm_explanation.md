# 路径规划层论文算法逻辑说明

本文档按路径规划层真实运行的逻辑顺序说明：每个步骤使用了哪篇论文的算法思想，以及关键公式的意义和作用。

## 1. 问题建模

这一层不是直接做动力学控制，而是生成满足 3-DOF 可跟踪性的覆盖路径。基础模型来自 `IROS2017a.pdf` 的 Dubins 覆盖思想，再结合 USV 的矩形覆盖模型。

关键公式：

```text
Delta = wf * (1 - rho)
```

`wf` 是矩形覆盖足迹宽度，`rho` 是重叠率。这个公式决定相邻覆盖条带中心线间距，作用是保证矩形足迹之间没有漏扫。

```text
H >= Rmin + lf / 2 + d_safe
```

`H` 是条带端部转向缓冲区。`Rmin` 保证能转弯，`lf / 2` 保证矩形足迹前后长度不越界，`d_safe` 保证安全余量。

## 2. 区域分解

区域分解主要来自两篇论文：

- `Joint-optimized coverage path planning framework for USV-assisted offshore bathymetric mapping From theory to practice.pdf` 用于“按支撑线/跨度最小”分解区域，目标是让每个子区域更适合往复式覆盖，并减少转弯。
- `s41598-025-20978-8.pdf` 用于凹区域处理，通过识别凹点，把复杂区域分成更容易覆盖的凸子区域。

核心思想不是单纯把区域切小，而是切成“覆盖代价低”的区域。这里的重点量是区域跨度 `B`，也就是某个覆盖方向下需要横向扫过的宽度。跨度越小，条带数量越少，转弯越少。

可采用的分解目标：

```text
min sum(B_i) + epsilon * triangle_penalty
```

`sum(B_i)` 控制总转弯次数，`triangle_penalty` 避免切出很瘦、很难让 USV 转弯的小三角区域。

## 3. 子区域覆盖模式生成

这一步主要来自 `Joint-optimized coverage path planning framework for USV-assisted offshore bathymetric mapping From theory to practice.pdf` 和 `IROS2017a.pdf`。

每个区域不只生成一条覆盖路径，而是生成多种候选覆盖模式：不同扫掠方向、不同入口、不同出口、不同往返顺序。这样后续 TSP 不是访问一个“点”，而是访问一个“带方向和覆盖路径的区域”。

每个候选模式的代价可以写成：

```text
C_pattern = L_sweep / v_ref + A_turn / r_max + L_dubins_turn / v_turn
```

`L_sweep / v_ref` 表示直线覆盖时间，`A_turn / r_max` 表示转向角带来的时间，`L_dubins_turn / v_turn` 表示条带之间满足最小转弯半径的连接代价。

这里 `IROS2017a.pdf` 的贡献很关键：它把每个覆盖通道看成有方向的节点，例如“从下往上扫”和“从上往下扫”是两个不同状态。路径规划层也保留入口位姿和出口位姿，因为 USV 有转弯半径，不能像普通机器人那样原地换向。

## 4. 负载均衡

多艇分配主要用 `Multi-UAV_Coverage_Path_Planning_Based_on_Balanced_Graph_Partitioning.pdf`。

它的核心是先把区域变成图：节点是子区域，边表示两个区域相邻或容易连接。然后每个节点有权重，权重不是面积，而是预计完成时间。

区域权重公式可以抽象为：

```text
W(region) = min_theta T_cover(region, theta)
```

其中：

```text
T_cover = L_total / v_ref + N_turn * T_turn
```

这个公式的意义是：同样面积的区域，如果形状更窄长、转弯更多，它的工作量就更大。用时间当权重，比用面积更适合无人艇覆盖任务。

负载均衡目标是：

```text
min max_i W_i
```

也就是尽量让最慢完成的那艘艇更快完成。实际实现里还会加一个负载差约束，例如每艘艇工作量与平均值偏差不超过 10%。

## 5. 改进 A* 连接优化

这一步主要来自 `1802.03221v1.pdf`，并吸收 `Efficient_Coverage_Path_Planning_and_Underwater_Topographic_Mapping_of_an_USV_Based_on_A-Improved_Bio-Inspired_Neural_Network.pdf` 中“短路径 + 少转弯”的局部选择思想。

普通 A* 只关心距离：

```text
f(n) = g(n) + h(n)
```

无人艇版本不能只看距离，因为靠近障碍、边界或频繁转弯都不好。所以边代价改成：

```text
edge_cost = distance * safety_weight + heading_penalty + curvature_penalty
```

`1802.03221v1.pdf` 里的安全权重思想是：如果候选节点附近不可通行网格越多，代价越高。论文中使用类似指数增长的邻域安全权重：

```text
w = 1 + 0.5 * 2^(n - 1)
```

`n` 表示邻域内危险或不可通行单元数量。意义是让路径主动远离障碍，而不是贴边走最短路。

论文还引入目标引导启发项：

```text
h(n) = distance(n, goal) * p(n)
p(n) = 3 / (4 - sin(theta))
```

`theta` 表示当前节点相对目标方向的夹角。这个公式的作用是让 A* 更偏向朝目标方向扩展，减少无意义搜索。

路径规划层会进一步加入转向代价：

```text
heading_penalty = w_theta * abs(delta_psi)
```

这个来自 A-IBINN 论文中的局部目标选择思想：不仅选未覆盖区域，也优先选转角小的下一个目标。

## 6. 单艇区域间 TSP-CPP

这是路径规划层最难的部分，主要来自 `Joint-optimized coverage path planning framework for USV-assisted offshore bathymetric mapping From theory to practice.pdf`、`IROS2017a.pdf` 和 `s41598-025-20978-8.pdf`。

普通 TSP 是访问点；这里不是。每个区域都有多个候选覆盖模式，每个模式有入口、出口、航向和内部覆盖路径。所以它更像 GTSP/ATSP：每个区域是一组候选路径，必须从每组里选一个。

候选选择公式来自 joint-optimized 那篇论文，可改写为：

```text
cost(candidate_j) =
    C_inside(candidate_j)
  + C_connect(prev_exit, candidate_entry)
  + C_lookahead(candidate_exit, next_region)
```

原论文多用欧氏距离，路径规划层把连接代价替换为 Dubins 代价：

```text
C_connect = DubinsLength(q_prev_exit, q_entry, Rmin)
```

这样才能体现无人艇最小转弯半径和入口/出口航向约束。

`s41598-025-20978-8.pdf` 的作用主要在后优化：区域访问顺序初始生成后，用 2-opt/3-opt 或 ACO 类方法交换访问顺序。每次交换后，不只是重算区域顺序，还要重新选择每个区域的覆盖方向、入口和出口。

## 7. 短路径和小转弯角联合优化

这个目标贯穿 A*、覆盖模式选择和 TSP 后优化。它融合了 `Efficient_Coverage_Path_Planning_and_Underwater_Topographic_Mapping_of_an_USV_Based_on_A-Improved_Bio-Inspired_Neural_Network.pdf` 的转角惩罚、`IROS2017a.pdf` 的 Dubins 转弯代价，以及 `Joint-optimized coverage path planning framework for USV-assisted offshore bathymetric mapping From theory to practice.pdf` 的区域内外联合优化。

总目标可以写成：

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

## 8. 输出到 3-DOF 跟踪层

最后输出不是离散节点，而是可被 3-DOF/NMPC 跟踪的参考轨迹。

每段路径必须满足：

```text
kappa <= 1 / Rmin
```

其中 `kappa` 是曲率。这个约束来自 `IROS2017a.pdf` 的 Dubins 车辆思想，也是无人艇转弯半径约束的几何表达。

最终路径由三部分组成：

- 区域内直线覆盖条带。
- 条带间 Dubins/Bezier 可行转弯。
- 区域间 Dubins 可行连接。

这样规划层生成的是“覆盖完整、负载均衡、少转弯、能被 3-DOF 模型跟踪”的路径。
