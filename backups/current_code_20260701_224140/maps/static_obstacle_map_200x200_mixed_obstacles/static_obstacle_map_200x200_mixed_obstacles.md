# static_obstacle_map_200x200_mixed_obstacles

## 用途

该地图是 `200m x 200m` 多形态静态障碍全覆盖路径规划实验环境资产。文件只描述任务区域和静态障碍物，不包含无人艇初始状态、覆盖路径、轨迹参考或仿真结果。

## 任务区域

- 区域类型：矩形水域。
- 坐标范围：`[0, 200] x [0, 200]`。
- 单位：`m`。
- 原点：左下角 `(0, 0)`。

## 覆盖与运动约束

- 覆盖足迹：矩形。
- 覆盖足迹长度：`length_lf = 4.0 m`。
- 覆盖足迹宽度：`width_wf = 2.0 m`。
- 无人艇模型：`3-DOF`。
- 最小转弯半径：`Rmin = 2.0 m`。
- 推荐无人艇数量：`5`。
- 推荐重叠率：`rho = 0.2`。
- 推荐安全距离：`d_safe = 1.0 m`。

## 静态障碍物

| ID | 类型 | 参数 |
| --- | --- | --- |
| `rect_platform_sw_01` | rectangle | center=`(32.0, 34.0)`, size=`16.0 x 12.0`, yaw=`0 deg` |
| `rotated_dock_sw_01` | rectangle | center=`(70.0, 38.0)`, size=`30.0 x 5.0`, yaw=`-25 deg` |
| `ellipse_reef_s_01` | ellipse | center=`(116.0, 36.0)`, radii=`(13.0, 5.0)`, yaw=`18 deg` |
| `circle_reef_s_01` | circle | center=`(148.0, 48.0)`, radius=`6.0` |
| `circle_buoy_s_02` | circle | center=`(168.0, 30.0)`, radius=`3.5` |
| `l_shape_island_w_01` | polygon/L-shape | vertices=`(24,88) -> (56,88) -> (56,102) -> (42,102) -> (42,130) -> (24,130)` |
| `u_shape_island_c_01` | polygon/U-shape | vertices=`(82,82) -> (124,82) -> (124,96) -> (100,96) -> (100,118) -> (124,118) -> (124,132) -> (82,132) -> (82,118) -> (90,118) -> (90,96) -> (82,96)` |
| `rotated_barrier_e_01` | rectangle | center=`(158.0, 102.0)`, size=`36.0 x 6.0`, yaw=`32 deg` |
| `triangle_shoal_e_01` | polygon/triangle | vertices=`(174,124) -> (194,134) -> (180,150)` |
| `ellipse_reef_n_01` | ellipse | center=`(54.0, 162.0)`, radii=`(18.0, 7.0)`, yaw=`-12 deg` |
| `trapezoid_island_n_01` | polygon/trapezoid | vertices=`(96,152) -> (132,146) -> (142,170) -> (104,180)` |
| `irregular_island_ne_01` | polygon/irregular | vertices=`(154,156) -> (170,150) -> (186,160) -> (182,178) -> (162,184) -> (148,172)` |
| `rock_circle_01` | circle | center=`(38.0, 72.0)`, radius=`3.0` |
| `rock_circle_02` | circle | center=`(52.0, 68.0)`, radius=`2.5` |
| `rock_ellipse_03` | ellipse | center=`(138.0, 74.0)`, radii=`(5.0, 2.5)`, yaw=`45 deg` |
| `rock_polygon_04` | polygon/irregular | vertices=`(66,134) -> (74,130) -> (82,138) -> (78,148) -> (68,146)` |

## 设计说明

- 障碍类型全部来自当前地图加载器支持的类型：`rectangle`、`ellipse`、`circle`、`polygon`。
- 地图包含规则障碍、旋转障碍、曲边障碍、凹多边形障碍和不规则多边形障碍。
- 障碍物不贴边，边界附近保留较大自由水域，便于后续算法生成全局覆盖路径。
- 多个小型礁石以独立障碍物表达，不使用未支持的复合障碍类型。

## 不包含内容

- 不包含无人艇初始状态。
- 不包含动态障碍物。
- 不包含覆盖路径。
- 不包含轨迹参考。
- 不包含仿真结果。

## 推荐输出目录

后续如果运行实验，推荐输出目录为：

```text
outputs/static_obstacle_map_200x200_mixed_obstacles_usv5_footprint4x2_rmin2/
```
