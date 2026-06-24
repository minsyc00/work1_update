# 50x50 大地图 USV1/USV2 覆盖效果修复计划

## Summary
当前 `outputs/static_obstacle_map_50x50_simple_usv3_footprint4x2_rmin1/paper_style_region_tsp/09_constraint_validation.png` 中 USV1/USV2 效果差，主要原因不是绘图问题，而是多个右侧大区域被选成了 `*_compressed` 单线扫描模式，导致右侧自由空间没有被完整往复式覆盖。报告显示 `coverage_fraction=0.7256`、`residual_count=11`、USV2 跳过 `perf_region_4`，说明区域模式筛选、负载权重和残差闭环需要收紧。

## Key Changes
- 不在本阶段重新跑完整 50x50 实验，但修复代码，使下一次实验不再把大区域压成单线覆盖。
- `*_compressed` 只允许用于单条 footprint 能覆盖的小/窄区域；大区域不得用单线 compressed 替代完整 boustrophedon 扫描。
- 每个 `RegionCoveragePattern` 增加区域覆盖率估计，排序、预筛选和 TSP 候选排序必须先满足覆盖率，再比较长度、转弯、重复路径和入口代价。
- 大地图预筛选必须保留每个区域的高覆盖候选，不能因为轻量评分短而丢掉完整扫描候选。
- 区域图节点权重改用完整覆盖工作量，避免低覆盖 compressed pattern 让负载均衡低估右侧区域工作量。
- 跳过区域必须反映到报告状态：`skipped_regions` 非空或覆盖率不达标时，报告和 metadata 标记为 incomplete。
- 残差补扫改为目标驱动：在 cycle 预算内持续补扫到 `target_coverage_fraction`，而不是追加若干残差后仍报告 success。

## Test Plan
- 单元测试：大区域不得生成低覆盖 compressed pattern，窄区域仍可使用 compressed pattern。
- 单元测试：大地图预筛选不会删除每个区域的最高覆盖候选。
- 单元测试：区域图节点权重不因低覆盖候选而显著低估区域工作量。
- 回归测试：现有论文式小图测试继续通过。

## Assumptions
- 覆盖正确性优先于路径长度；不允许为了速度选择低覆盖单线模式。
- `target_coverage_fraction` 默认继续使用 `0.99`。
- 当前阶段仍处理静态障碍离线路径规划；动态障碍不进入本次修复。
- 本次不重新跑完整 50x50 实验，只跑轻量和单元测试。
