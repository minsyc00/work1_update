# 各实验入口完整命令参考

> 环境：Windows + Anaconda (`D:\anaconda3\envs\pytorch_gpu\`)
> 项目根目录：`d:\code\work1_update\`
> Python 路径前缀：`D:\anaconda3\envs\pytorch_gpu\python.exe`

---

## 目录

1. [地图清单](#1-地图清单)
2. [入口脚本总览](#2-入口脚本总览)
3. [入口①：论文式区域 TSP 实验（主要入口）](#3-入口论文式区域-tsp-实验主要入口)
4. [入口②：全算法过程实验](#4-入口全算法过程实验)
5. [入口③：路径规划层 Demo](#5-入口路径规划层-demo)
6. [入口④：静态障碍物路径规划 Demo](#6-入口静态障碍物路径规划-demo)
7. [入口⑤：闭环仿真 Demo](#7-入口闭环仿真-demo)
8. [入口⑥：基础 Demo（覆盖+控制）](#8-入口基础-demo覆盖控制)
9. [TSP 求解器对比速查](#9-tsp-求解器对比速查)
10. [输出文件解读](#10-输出文件解读)

---

## 1. 地图清单

| 地图 ID | 尺寸 | 障碍物 | 推荐 USV 数 | JSON 路径 |
|---|---|---|---|---|
| `static_obstacle_map_10x10_rect_obstacle` | 10×10m | 1 矩形 (2×2m) | 1 | `maps\static_obstacle_map_10x10_rect_obstacle\...json` |
| `static_obstacle_map_15x15_rect_triangle_small` | 15×15m | 1 矩形 (1.4×1.2m) + 1 三角形 | 2 | `maps\static_obstacle_map_15x15_rect_triangle_small\...json` |
| `static_obstacle_map_20x20_two_obstacles` | 20×20m | 1 矩形 (2×2.5m) + 1 椭圆 (r=1.4,0.8m) | 2 | `maps\static_obstacle_map_20x20_two_obstacles\...json` |
| `static_obstacle_map_50x50_simple` | 50×50m | 4 个：矩形+椭圆+L形多边形+旋转矩形 | 3 | `maps\static_obstacle_map_50x50_simple\...json` |

地图 JSON 中可提取的参数：

| 字段 | 说明 |
|---|---|
| `mission_area.length_x / length_y` | 任务区域尺寸 (m) |
| `coverage_footprint.length_lf / width_wf` | 覆盖足迹 (m) |
| `motion_constraints.min_turn_radius` | 最小转弯半径 (m) |
| `notes.recommended_usv_count` | 推荐 USV 数量 |
| `notes.recommended_overlap_ratio` | 推荐覆盖重叠率 |
| `notes.recommended_d_safe` | 推荐安全距离 (m) |

---

## 2. 入口脚本总览

| 入口 | 脚本 | 特点 |
|---|---|---|
| ① | `run_paper_style_region_tsp_experiment.py` | **主要入口**：区域分解→模式生成→图构建→TSP→残余回填→可视化输出 |
| ② | `run_full_algorithm_experiment.py` | 全算法步骤图（含 TSP 记录），输出 GIF |
| ③ | `path_planning_layer_demo.py` | 无障碍物矩形区域，纯路径规划层测试 |
| ④ | `static_obstacle_path_planning_demo.py` | 硬编码障碍物（无 CLI 地图参数） |
| ⑤ | `closed_loop_simulation.py` | 闭环仿真 + 动态障碍物 + GIF 动画 |
| ⑥ | `demo.py` | 基础覆盖规划+NMPC，无 CLI 参数 |

---

## 3. 入口①：论文式区域 TSP 实验（主要入口）

**脚本**：`examples\run_paper_style_region_tsp_experiment.py`

**CLI 参数**：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--map` | str | 20x20_two_obstacles | 地图 JSON |
| `--outputs-root` | str | `outputs\` | 输出根目录 |
| `--dpi` | int | 140 | 输出图片 DPI |
| `--rmin` | float | 地图默认值 (2.0) | 覆盖最小转弯半径 |
| `--tsp-solver` | choice | `deterministic` | `deterministic` / `aco` / `fa3aco` |
| `--aco-ants` | int | 30 | ACO 蚂蚁数 |
| `--aco-iterations` | int | 80 | ACO 迭代次数 |
| `--aco-seed` | int | 42 | ACO 随机种子 |
| `--tsp-2opt-iterations` | int | 8 | 2-opt 最大迭代次数 |
| `--performance-profile` | choice | `balanced` | `balanced` / `shortest` / `low-repeat` |
| `--target-coverage` | float | 0.99 | 目标覆盖率 |
| `--usv-count` | int | 自动(>=50m用推荐值) | 覆盖 USV 数量 |
| `--run-parameter-sweep` | flag | false | 运行参数扫描后渲染最优 |
| `--monitor-stages` | flag | false | 输出每阶段耗时 JSON |
| `--no-render` | flag | false | 仅计算，不生成图片 |

---

### 3.1 10×10 矩形障碍物地图（1 USV）

```powershell
# deterministic
D:\anaconda3\envs\pytorch_gpu\python.exe examples\run_paper_style_region_tsp_experiment.py --map maps\static_obstacle_map_10x10_rect_obstacle\static_obstacle_map_10x10_rect_obstacle.json --rmin 0.5 --tsp-solver deterministic --usv-count 1 --dpi 120

# aco
D:\anaconda3\envs\pytorch_gpu\python.exe examples\run_paper_style_region_tsp_experiment.py --map maps\static_obstacle_map_10x10_rect_obstacle\static_obstacle_map_10x10_rect_obstacle.json --rmin 0.5 --tsp-solver aco --aco-ants 40 --aco-iterations 120 --aco-seed 42 --usv-count 1 --dpi 120

# fa3aco
D:\anaconda3\envs\pytorch_gpu\python.exe examples\run_paper_style_region_tsp_experiment.py --map maps\static_obstacle_map_10x10_rect_obstacle\static_obstacle_map_10x10_rect_obstacle.json --rmin 0.5 --tsp-solver fa3aco --aco-ants 40 --aco-iterations 120 --aco-seed 42 --usv-count 1 --dpi 120
```

---

### 3.2 15×15 矩形+三角形地图（2 USV）

```powershell
# deterministic
D:\anaconda3\envs\pytorch_gpu\python.exe examples\run_paper_style_region_tsp_experiment.py --map maps\static_obstacle_map_15x15_rect_triangle_small\static_obstacle_map_15x15_rect_triangle_small.json --rmin 0.5 --tsp-solver deterministic --dpi 120

# aco
D:\anaconda3\envs\pytorch_gpu\python.exe examples\run_paper_style_region_tsp_experiment.py --map maps\static_obstacle_map_15x15_rect_triangle_small\static_obstacle_map_15x15_rect_triangle_small.json --rmin 0.5 --tsp-solver aco --aco-ants 40 --aco-iterations 120 --aco-seed 42 --dpi 120

# fa3aco
D:\anaconda3\envs\pytorch_gpu\python.exe examples\run_paper_style_region_tsp_experiment.py --map maps\static_obstacle_map_15x15_rect_triangle_small\static_obstacle_map_15x15_rect_triangle_small.json --rmin 0.5 --tsp-solver fa3aco --aco-ants 40 --aco-iterations 120 --aco-seed 42 --dpi 120

# fa3aco + 性能优化 + 阶段监控
D:\anaconda3\envs\pytorch_gpu\python.exe examples\run_paper_style_region_tsp_experiment.py --map maps\static_obstacle_map_15x15_rect_triangle_small\static_obstacle_map_15x15_rect_triangle_small.json --rmin 0.5 --tsp-solver fa3aco --aco-ants 40 --aco-iterations 120 --aco-seed 42 --performance-profile low-repeat --target-coverage 1.0 --monitor-stages --dpi 150

# 参数扫描
D:\anaconda3\envs\pytorch_gpu\python.exe examples\run_paper_style_region_tsp_experiment.py --map maps\static_obstacle_map_15x15_rect_triangle_small\static_obstacle_map_15x15_rect_triangle_small.json --rmin 0.5 --tsp-solver fa3aco --aco-ants 40 --aco-iterations 120 --aco-seed 42 --run-parameter-sweep --dpi 120

# 仅报告不渲染（快速验证）
D:\anaconda3\envs\pytorch_gpu\python.exe examples\run_paper_style_region_tsp_experiment.py --map maps\static_obstacle_map_15x15_rect_triangle_small\static_obstacle_map_15x15_rect_triangle_small.json --rmin 0.5 --tsp-solver fa3aco --aco-ants 40 --aco-iterations 120 --aco-seed 42 --no-render
```

---

### 3.3 20×20 双障碍物地图（2 USV）

```powershell
# deterministic
D:\anaconda3\envs\pytorch_gpu\python.exe examples\run_paper_style_region_tsp_experiment.py --map maps\static_obstacle_map_20x20_two_obstacles\static_obstacle_map_20x20_two_obstacles.json --rmin 0.8 --tsp-solver deterministic --dpi 120

# aco
D:\anaconda3\envs\pytorch_gpu\python.exe examples\run_paper_style_region_tsp_experiment.py --map maps\static_obstacle_map_20x20_two_obstacles\static_obstacle_map_20x20_two_obstacles.json --rmin 0.8 --tsp-solver aco --aco-ants 50 --aco-iterations 150 --aco-seed 42 --dpi 120

# fa3aco
D:\anaconda3\envs\pytorch_gpu\python.exe examples\run_paper_style_region_tsp_experiment.py --map maps\static_obstacle_map_20x20_two_obstacles\static_obstacle_map_20x20_two_obstacles.json --rmin 0.8 --tsp-solver fa3aco --aco-ants 50 --aco-iterations 150 --aco-seed 42 --dpi 120
```

---

### 3.4 50×50 大地图（3 USV，自动检测）

```powershell
# deterministic（自动使用3 USV）
D:\anaconda3\envs\pytorch_gpu\python.exe examples\run_paper_style_region_tsp_experiment.py --map maps\static_obstacle_map_50x50_simple\static_obstacle_map_50x50_simple.json --rmin 1.0 --tsp-solver deterministic --dpi 120 --tsp-2opt-iterations 4

# aco
D:\anaconda3\envs\pytorch_gpu\python.exe examples\run_paper_style_region_tsp_experiment.py --map maps\static_obstacle_map_50x50_simple\static_obstacle_map_50x50_simple.json --rmin 1.0 --tsp-solver aco --aco-ants 40 --aco-iterations 100 --aco-seed 1234 --dpi 120

# fa3aco
D:\anaconda3\envs\pytorch_gpu\python.exe examples\run_paper_style_region_tsp_experiment.py --map maps\static_obstacle_map_50x50_simple\static_obstacle_map_50x50_simple.json --rmin 1.0 --tsp-solver fa3aco --aco-ants 40 --aco-iterations 100 --aco-seed 1234 --dpi 120

# fa3aco + 手动指定4 USV
D:\anaconda3\envs\pytorch_gpu\python.exe examples\run_paper_style_region_tsp_experiment.py --map maps\static_obstacle_map_50x50_simple\static_obstacle_map_50x50_simple.json --rmin 1.0 --tsp-solver fa3aco --aco-ants 40 --aco-iterations 100 --aco-seed 1234 --usv-count 4 --dpi 120
```

---

## 4. 入口②：全算法过程实验

**脚本**：`examples\run_full_algorithm_experiment.py`

**CLI 参数**：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--map` | str | 20x20_two_obstacles | 地图 JSON |
| `--outputs-root` | str | `outputs\` | 输出根目录 |
| `--dpi` | int | 140 | 输出图片 DPI |
| `--gif-fps` | int | 4 | 算法过程 GIF 帧率 |
| `--tsp-solver` | choice | `deterministic` | `deterministic` / `aco` / `fa3aco` |
| `--aco-ants` | int | 30 | ACO 蚂蚁数 |
| `--aco-iterations` | int | 80 | ACO 迭代次数 |
| `--aco-seed` | int | 42 | ACO 随机种子 |

> **注意**：此脚本使用硬编码的 `build_two_usv_fleet()`（2 USV），起点固定为 (2,2) 和 (2,18)。

```powershell
# deterministic（20×20 双障碍物）
D:\anaconda3\envs\pytorch_gpu\python.exe examples\run_full_algorithm_experiment.py --map maps\static_obstacle_map_20x20_two_obstacles\static_obstacle_map_20x20_two_obstacles.json --tsp-solver deterministic --dpi 120 --gif-fps 4

# fa3aco
D:\anaconda3\envs\pytorch_gpu\python.exe examples\run_full_algorithm_experiment.py --map maps\static_obstacle_map_20x20_two_obstacles\static_obstacle_map_20x20_two_obstacles.json --tsp-solver fa3aco --aco-ants 40 --aco-iterations 120 --aco-seed 42 --dpi 120 --gif-fps 4

# 15×15 矩形+三角形
D:\anaconda3\envs\pytorch_gpu\python.exe examples\run_full_algorithm_experiment.py --map maps\static_obstacle_map_15x15_rect_triangle_small\static_obstacle_map_15x15_rect_triangle_small.json --tsp-solver fa3aco --aco-ants 40 --aco-iterations 120 --aco-seed 42 --dpi 120 --gif-fps 4

# 50×50 大地图（注意：硬编码为 2 USV，地图推荐 3 USV）
D:\anaconda3\envs\pytorch_gpu\python.exe examples\run_full_algorithm_experiment.py --map maps\static_obstacle_map_50x50_simple\static_obstacle_map_50x50_simple.json --tsp-solver fa3aco --aco-ants 40 --aco-iterations 100 --aco-seed 42 --dpi 120 --gif-fps 3
```

**输出目录**：`outputs\<map_id>_usv2_footprint4x2_rmin2p0\algorithm_steps\`

**重点文件**：
```
00_map_overview.png
01_region_decomposition.png
02_coverage_patterns.png            (候选模式)
03_region_graph.png
04_balanced_assignment.png
05_tsp_solution_agent_0.png         (每个 USV 的巡游)
05_tsp_solution_agent_1.png
06_residual_backfill.png
07_final_tour_paths.png
algorithm_process.gif               (全过程动画)
algorithm_experiment_report.json    (含 TSP 记录)
```

---

## 5. 入口③：路径规划层 Demo

**脚本**：`examples\path_planning_layer_demo.py`

**场景**：64×24m 无静态障碍物矩形区域，可变数量 USV。

**CLI 参数**：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--agents` | int | 4 | USV 数量 |
| `--tsp-solver` | choice | `deterministic` | `deterministic` / `aco` / `fa3aco` |
| `--aco-ants` | int | 30 | ACO 蚂蚁数 |
| `--aco-iterations` | int | 80 | ACO 迭代次数 |
| `--aco-seed` | int | 42 | ACO 随机种子 |
| `--three-opt` | flag | false | 启用确定性 3-opt（仅对 deterministic 有意义） |

```powershell
# deterministic（2 USV）
D:\anaconda3\envs\pytorch_gpu\python.exe examples\path_planning_layer_demo.py --agents 2 --tsp-solver deterministic

# deterministic + 3-opt
D:\anaconda3\envs\pytorch_gpu\python.exe examples\path_planning_layer_demo.py --agents 3 --tsp-solver deterministic --three-opt

# aco（2 USV）
D:\anaconda3\envs\pytorch_gpu\python.exe examples\path_planning_layer_demo.py --agents 2 --tsp-solver aco --aco-ants 40 --aco-iterations 120 --aco-seed 42

# fa3aco（4 USV）
D:\anaconda3\envs\pytorch_gpu\python.exe examples\path_planning_layer_demo.py --agents 4 --tsp-solver fa3aco --aco-ants 40 --aco-iterations 120 --aco-seed 42

# fa3aco（8 USV，大规模测试）
D:\anaconda3\envs\pytorch_gpu\python.exe examples\path_planning_layer_demo.py --agents 8 --tsp-solver fa3aco --aco-ants 30 --aco-iterations 80 --aco-seed 42
```

**终端输出解读**：
```
algorithm: paper_fusion_planner
status: paper_fusion
regions: 8
coverage_fraction: 0.987654
requested_tsp_solver: fa3aco
effective_tsp_solver: fa3aco        ← 看这一行！
tsp_solver_status: success          ← success = ACO 成功
load_imbalance_ratio: 0.023456
planning_time: 2.345678s
agent 0: regions=4, segments=18, length=112.34, turn=3.45, max_kappa=0.250, ref_samples=234
agent 1: regions=4, segments=16, length=108.67, turn=3.12, max_kappa=0.250, ref_samples=218
```

如果看到：
```
effective_tsp_solver: deterministic_fallback    ← ACO 回退！
tsp_solver_status: failed
```
说明 ACO/FA3ACO 生成的最优解未通过最终约束验证（运动学不可行/碰撞等），系统自动回退到确定性求解器。

---

## 6. 入口④：静态障碍物路径规划 Demo

**脚本**：`examples\static_obstacle_path_planning_demo.py`

**场景**：48×18m，硬编码 4 个障碍物（矩形码头+圆形浮标+椭圆礁石+多边形岩石），3 USV。

**无 CLI 参数**（参数硬编码在脚本中）。

```powershell
D:\anaconda3\envs\pytorch_gpu\python.exe examples\static_obstacle_path_planning_demo.py
```

> 如需该入口也支持 `--tsp-solver`，需要修改脚本添加 argparse。

---

## 7. 入口⑤：闭环仿真 Demo

**脚本**：`examples\closed_loop_simulation.py`

**场景**：52×24m，3 USV，第1层快速覆盖规划 + 闭环 NMPC 执行 + 2 个穿越动态障碍物。

**CLI 参数**：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--total-time` | float | 28.0 | 仿真总时长 (s) |
| `--fps` | int | 6 | GIF 帧率 |
| `--output` | str | `outputs\usv_swarm_closed_loop.gif` | 输出 GIF 路径 |

```powershell
# 默认参数
D:\anaconda3\envs\pytorch_gpu\python.exe examples\closed_loop_simulation.py

# 更长仿真 + 更高帧率
D:\anaconda3\envs\pytorch_gpu\python.exe examples\closed_loop_simulation.py --total-time 45.0 --fps 10 --output outputs\sim_45s.gif

# 短仿真快速验证
D:\anaconda3\envs\pytorch_gpu\python.exe examples\closed_loop_simulation.py --total-time 15.0 --fps 4 --output outputs\quick_test.gif
```

> **注意**：该入口使用第1层 `plan_global_coverage()`（无障碍物带状覆盖）。如需障碍物感知规划 + 闭环仿真，需自行组合 `PathPlanningLayer` + `SwarmRuntime`。

---

## 8. 入口⑥：基础 Demo（覆盖+控制）

**脚本**：`examples\demo.py`

**场景**：60×24m，3 USV，第1层覆盖 + 单步 NMPC 控制。

**无 CLI 参数**。

```powershell
D:\anaconda3\envs\pytorch_gpu\python.exe examples\demo.py
```

**输出示例**：
```
strips: 7
makespan: 24.56s
agent 0: strips=(0, 2), ref_samples=156
  cmd=(thrust=1.23, yaw=0.45) mode=nominal margin=2.89
agent 1: strips=(3, 4), ref_samples=98
  cmd=(thrust=1.15, yaw=-0.32) mode=nominal margin=2.91
agent 2: strips=(5, 6), ref_samples=102
  cmd=(thrust=1.08, yaw=0.18) mode=nominal margin=2.87
coverage fraction after one update: 0.023
```

---

## 9. TSP 求解器对比速查

| 求解器 | `--tsp-solver` | 算法 | ACO 特有参数 |
|---|---|---|---|
| **确定性** | `deterministic` | A\* 种子排序 → 2-opt → 3-opt | 无 |
| **ACO** | `aco` | 标准蚁群 + 确定性回退 | `--aco-ants` `--aco-iterations` `--aco-seed` |
| **FA³ACO** | `fa3aco` | 分数阶记忆 ACO + 自适应蒸发 + 3-opt + 确定性回退 | 同上 |

**关键区别**：

| 特性 | deterministic | aco | fa3aco |
|---|---|---|---|
| 搜索范围 | 局部（2-opt 邻域） | 全局（蚁群采样） | 全局 + 记忆 |
| 解的质量 | 依赖初始 A\* 排序 | 通常优于 deterministic | 大区域数时最优 |
| 速度 | 最快 | 较慢（I×A 次评估） | 最慢（+ 3-opt） |
| 可复现性 | 完全确定 | 依赖 `--aco-seed` | 依赖 `--aco-seed` |
| 大问题适用性 | 好（自适应缩减迭代） | 中（随区域数增加） | 优（记忆避免局部最优） |

**回退机制**：ACO/FA³ACO 选出的巡游方案在段组装阶段会经过完整的障碍物/运动学验证。若任何区域间过渡无法生成可行的 `PathSegmentSpec`（即 `kinematic_feasible == "false"` 或存在碰撞/越界），则回退到确定性求解器，并在终端输出中标记：

```
effective_tsp_solver: deterministic_fallback
tsp_solver_status: failed
```

---

## 10. 输出文件解读

### 10.1 入口① 输出目录

```
outputs\<map_id>_usv<N>_footprint<L>x<W>_rmin<R>p<X>\paper_style_region_tsp\
│
├── 00_input_overview.png               ← 输入地图总览
├── 01_obstacle_field.png               ← 膨胀后的障碍物场
├── 02_region_decomposition.png         ← 自由空间区域分解
├── 03_coverage_patterns.png            ← 候选覆盖模式（每个区域）
├── 04_region_graph.png                 ← 区域邻接图
├── 05_balanced_assignment.png          ← 负载均衡分配
├── 06_tsp_solution_agent_0.png         ← USV 0 的 TSP 巡游方案
├── 06_tsp_solution_agent_1.png         ← USV 1 的 TSP 巡游方案
├── ...
├── 07_residual_backfill.png            ← 残余覆盖回填
├── 08_final_region_tsp_coverage_path.png  ← 最终覆盖路径 ★
├── 09_constraint_validation.png        ← 约束验证（碰撞/越界/曲率） ★
├── paper_style_region_tsp_report.json  ← 完整 JSON 报告 ★
└── sweep\                              ← 参数扫描子目录（如启用）
```

### 10.2 JSON 报告关键字段

```json
{
  "coverage_fraction": 0.987,
  "requested_tsp_solver": "fa3aco",
  "effective_tsp_solver": "fa3aco",
  "tsp_solver_status": "success",
  "tsp_node_count": "8",
  "coverage_endpoint_count": "24",
  "invalid_path_length": "0.000000",
  "out_of_bounds_segment_count": "0",
  "obstacle_collision_segment_count": "0",
  "kinematic_infeasible_segment_count": "0",
  "performance_profile": "balanced",
  "transition_length_ratio": 0.234,
  "repeat_transition_ratio": 0.012,
  "target_coverage_met": true,
  "constraint_ok": true,
  "agent_0": {
    "initial_order": ["region_3", "region_0", "region_1", "region_5"],
    "planned_region_order": "...",
    "2opt_improvements": 3,
    "final_metrics": { "length": 112.3, "turn_angle": 3.45, "objective": 145.6 }
  },
  "aco_metadata": {
    "aco_best_objective": 134.2,
    "aco_initial_objective": 178.9,
    "aco_iteration_count": 120,
    "aco_accepted_3opt_count": 5
  }
}
```

### 10.3 快速验证清单

运行完成后检查终端输出的这几行：

```
effective_tsp_solver: fa3aco           ← 应为请求的求解器
tsp_solver_status: success             ← 应为 success
invalid_path_length: 0.000000          ← 应为 0
out_of_bounds_segment_count: 0         ← 应为 0
obstacle_collision_segment_count: 0    ← 应为 0
kinematic_infeasible_segment_count: 0  ← 应为 0
```

如果任一项非零，查看 `09_constraint_validation.png` 定位问题段。
