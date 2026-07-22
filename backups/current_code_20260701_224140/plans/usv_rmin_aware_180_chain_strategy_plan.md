# Rmin/覆盖范围感知的 180 度掉头 Chain 策略计划

## Summary

目标是在论文式区域扫描中引入与无人艇最小转弯半径和矩形覆盖范围相关的 chain 构造策略。当前相邻扫描线间距为 `Delta = wf * (1 - rho)`，当 `Delta < 2 * Rmin` 时，相邻 pass 之间很难通过符合 3-DOF、动力学和最小转弯半径的 180 度掉头。新策略不再强制连接相邻 pass，而是按可掉头跨度生成交错 chain，使绝大多数扫描 pass 能通过合法 180 度掉头完成覆盖。

## Key Formula

- 覆盖条带间距：
  `Delta = wf * (1 - rho)`
- 180 度掉头所需横向跨度：
  `required_turn_span = 2 * Rmin + max(0.25 * wf, d_safe)`
- 交错扫描步长：
  `turn_stride = max(1, ceil(required_turn_span / Delta))`

当 `turn_stride = 1` 时，保持传统往复扫描。  
当 `turn_stride > 1` 时，生成交错 chain，例如 `s = 3` 时：

- `chain_0 = pass[0], pass[3], pass[6]...`
- `chain_1 = pass[1], pass[4], pass[7]...`
- `chain_2 = pass[2], pass[5], pass[8]...`

所有 pass 仍被覆盖，只是执行顺序从相邻往复变成按可掉头跨度分组。

## Implementation Plan

- 备份当前代码到 `backups/code_snapshot_<timestamp>_before_rmin_aware_180_chain`。
- 新增配置项：
  `enable_rmin_aware_chain_order`、`chain_turn_strategy`、`rmin_chain_turn_clearance_factor`、`rmin_chain_max_stride`、`rmin_chain_min_pass_length_factor`。
- 扩展 `OpenSweepChain.metadata`，记录 `chain_order_mode`、`turn_stride`、`required_turn_span`、`delta` 和 180 度掉头验证统计。
- 在 open-chain 构建前先生成 Rmin-aware pass groups。
- chain 内连接优先尝试 Dubins/Bezier 180 度掉头，失败后再调用 obstacle-aware repair。
- chain 间连接交给 chain-level TSP，不再伪装为区域内部 U-turn。
- 短 pass、障碍裁剪 pass 或边界不足 pass 不跳过整个 region，而是降级为 single-pass chain 或 residual chain。

## Report And Visualization

- 报告新增：
  `rmin_aware_chain_enabled`、`turn_stride_distribution`、`rmin_180_attempt_count`、`rmin_180_success_count`、`rmin_180_feasible_ratio`、`single_pass_chain_count`、`short_pass_residual_count`。
- `04_region_sweep_patterns.png` 标注 `Delta`、`Rmin`、`turn_stride`。
- `04_open_sweep_chain_tsp.png` 显示交错 chain、chain 内 180 度掉头和 chain 间 TSP 连接。

## Test Plan

- `Rmin=2, wf=2, rho=0.2` 时应得到 `turn_stride=3`，并且所有 pass 被分配到交错 chain。
- `Delta >= 2 * Rmin` 时 `turn_stride=1`，结果与普通往复扫描一致。
- chain 内所有 180 度掉头必须通过曲率、航向连续、yaw-rate、yaw-acceleration 和 3-DOF 验证。
- 障碍导致局部掉头失败时，只断开局部 chain，不跳过整个 region。
- 50x50 快速实验中，目标是 `open_chain_connected_count` 上升、`skipped_region_count` 下降、`cover_only_coverage_fraction` 提升，并保持最终违规计数为 0。

## Assumptions

- 当前阶段只针对 3-DOF 静态障碍覆盖规划。
- 不放宽边界、障碍、曲率、转弯半径或动态验证约束。
- 若某些狭窄 pass 无法满足 180 度掉头，则进入 single-pass/residual 处理，而不是静默丢弃。
