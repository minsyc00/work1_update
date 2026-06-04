# static_obstacle_map_50x50_simple

## 用途

该地图是多无人艇静态障碍全覆盖路径规划实验的初始环境资产。文件只描述任务区域和静态障碍物，不包含无人艇初始状态、路径规划结果、覆盖条带或仿真轨迹。

## 任务区域

- 区域类型：矩形水域。
- 坐标范围：`[0, 50] x [0, 50]`。
- 单位：`m`。
- 原点：左下角 `(0, 0)`。

## 覆盖与运动约束

- 覆盖足迹：矩形。
- 覆盖足迹长度：`length_lf = 4.0 m`。
- 覆盖足迹宽度：`width_wf = 2.0 m`。
- 无人艇模型：`3-DOF`。
- 最小转弯半径：`Rmin = 2.0 m`。
- 推荐重叠率：`rho = 0.2`。
- 推荐安全距离：`d_safe = 1.0 m`。

## 静态障碍物

| ID | 类型 | 参数 |
| --- | --- | --- |
| `rect_platform_01` | rectangle | center=`(13.0, 12.0)`, size=`5.0 x 6.0`, yaw=`0 deg` |
| `elliptic_reef_01` | ellipse | center=`(34.0, 13.0)`, radii=`(4.0, 2.0)`, yaw=`20 deg` |
| `l_shape_island_01` | polygon | vertices=`(13,28) -> (23,28) -> (23,32) -> (19,32) -> (19,40) -> (13,40)` |
| `rotated_barrier_01` | rectangle | center=`(40.0, 36.0)`, size=`7.0 x 2.0`, yaw=`-30 deg` |

## 不包含内容

- 不包含无人艇数量或初始位姿。
- 不包含动态障碍物。
- 不包含覆盖路径。
- 不包含轨迹参考。
- 不包含仿真结果。

## 输出目录规范

该地图对应的推荐实验输出目录为：

```text
D:\code\work1_update\outputs\static_obstacle_map_50x50_simple_usv3_footprint4x2_rmin2\
```

后续任何算法运行结果，例如 `path_plan.png`、`path_plan.gif`、`coverage_report.json`、`trajectory_refs.json`、`planning_metrics.json`、`simulation_log.json`，都应保存到该输出目录，而不是直接保存到 `outputs` 根目录。
