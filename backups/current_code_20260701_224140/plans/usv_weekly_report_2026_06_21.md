# 多无人艇覆盖路径规划项目周报

日期：2026-06-21

## 本周工作概述

本周主要围绕多无人艇静态障碍覆盖路径规划的 3-DOF 版本继续推进，重点解决复杂地图中“区域内部无法掉头导致大片区域不覆盖”的问题，并对 50m x 50m 大图实验进行了阶段监控和瓶颈定位。

## 已完成工作

1. 完成项目功能与流程总结文档，并根据当前研究重点去除了 6-DOF 相关表述，统一改为面向 3-DOF 无人艇模型、最小转弯半径约束和 NMPC 轨迹跟踪的说明。

2. 设计并实现了 `OpenSweepChain` 机制：当某个覆盖区块内部少数 U-turn 不可行时，不再直接判定整个区块不可覆盖，而是将扫描 pass 切分成若干最大连续可行扫描链，尽量保留可覆盖扫描线。

3. 新增了 `OpenSweepChain` 和 `OpenSweepBreak` 数据结构，并加入 open-chain 相关配置、报告字段和可视化诊断入口。

4. 新增了 open-chain 单元测试，验证了两类核心场景：U-turn 被障碍阻断时能切成多个 chain；cover pass 穿越障碍时能被识别为无效 pass，而不是污染有效扫描链。

5. 对 50x50 静态障碍大图进行了带 `--monitor-stages` 的完整运行尝试，并生成了本次超时运行的阶段诊断图和 JSON 摘要。

## 当前实验结果

50x50 大图实验配置为：

```text
map = static_obstacle_map_50x50_simple
USV = 3
footprint = 4m x 2m
Rmin = 1.0m
tsp_solver = deterministic
target_coverage = 0.99
```

本次实验没有跑到最终渲染阶段，1 小时后超时。但阶段监控显示：

1. `feasible_region_count` 从旧诊断的 `82` 提升到 `87`，说明 OpenSweepChain 已经改善了 sweep 阶段的区域可行性。
2. `open_chain_region_count=209`，`open_chain_count=1266`，`open_chain_break_count=1073`，说明大量原本因 U-turn 问题失败的扫描序列已经被切分成可分析的扫描链。
3. 区域 TSP 阶段仍是主要瓶颈，当前只访问了约 `3/87` 个分配区域，导致最终覆盖路径无法生成。
4. 当前问题已经从“区域没有进入可行候选”转移为“open-chain/region 之间缺少稳定可达连接和 TSP 选择机制”。

本次诊断输出：

```text
outputs/static_obstacle_map_50x50_simple_usv3_footprint4x2_rmin1/paper_style_region_tsp/latest_50x50_open_chain_timeout_summary.png
outputs/static_obstacle_map_50x50_simple_usv3_footprint4x2_rmin1/paper_style_region_tsp/latest_50x50_open_chain_timeout_summary.json
```

## 存在问题

1. 50x50 大图完整生成所有结果图片预计需要 60-90 分钟，当前实现仍可能超时。

2. OpenSweepChain 已能提升 sweep 阶段可行区域数量，但 chain 之间尚未形成高效稳定的可达图，导致 TSP 阶段跳过大量区域。

3. `open_chain_connected_count` 当前仍为 0，因为 chain 重连接被延后到 TSP 展开阶段，缺少一个显式的 chain-level reachability graph。

4. 区域分配仍主要基于区域权重，没有充分考虑 open-chain/region 间的可达性，导致部分无人艇被分配到大量后续无法访问的区域。

## 下周计划

1. 增加 chain-level reachability graph，在主区域 TSP 前先判断 open-chain 之间和 region 之间的可达性。

2. 将 OpenSweepChain 从“候选补救机制”升级为正式的局部 TSP-CPP 子问题，使无法内部掉头的扫描链能被外部可跟踪连接重新串起来。

3. 优化 50x50 大图运行时间，重点减少 `build_sweep_paths` 和区域 TSP 阶段的重复动态验证。

4. 增加部分渲染模式，只跑到区域分解和 sweep path 构建阶段即可更新 `00-04` 类诊断图，避免每次都完整跑 TSP。

5. 在 50x50 地图上重新验证覆盖率、skipped region 数量、open-chain 连接数量和最终路径约束。

