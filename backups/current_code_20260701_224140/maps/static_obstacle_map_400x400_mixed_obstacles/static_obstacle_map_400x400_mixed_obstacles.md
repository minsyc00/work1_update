# static_obstacle_map_400x400_mixed_obstacles

## 用途

该地图是 `400m x 400m` 多形态静态障碍全覆盖路径规划实验环境资产。文件只描述任务区域和静态障碍物，不包含无人艇初始状态、覆盖路径、轨迹参考或仿真结果。

本地图在 200x200 多形态地图的基础上扩大尺度，并加入一个较短、更平缓的 S 型弧形障碍 `s_curve_barrier_c_01`，用于测试现有区域分解、障碍膨胀和往复扫描生成逻辑对弯曲障碍的适配能力。

## 任务区域

- 区域类型：矩形水域。
- 坐标范围：`[0, 400] x [0, 400]`。
- 单位：`m`。
- 原点：左下角 `(0, 0)`。

## 覆盖与运动约束

- 覆盖足迹：矩形。
- 覆盖足迹长度：`length_lf = 4.0 m`。
- 覆盖足迹宽度：`width_wf = 2.0 m`。
- 无人艇模型：`3-DOF`。
- 最小转弯半径：`Rmin = 2.0 m`。
- 推荐无人艇数量：`8`。
- 推荐重叠率：`rho = 0.2`。
- 推荐安全距离：`d_safe = 1.0 m`。

## 静态障碍物

| ID | 类型 | 参数 |
| --- | --- | --- |
| `rect_platform_sw_01` | rectangle | center=`(45.0, 45.0)`, size=`28.0 x 22.0`, yaw=`0 deg` |
| `rotated_dock_sw_01` | rectangle | center=`(112.0, 50.0)`, size=`55.0 x 8.0`, yaw=`-22 deg` |
| `ellipse_reef_s_01` | ellipse | center=`(168.0, 70.0)`, radii=`(22.0, 8.0)`, yaw=`12 deg` |
| `circle_reef_se_01` | circle | center=`(356.0, 46.0)`, radius=`10.0` |
| `circle_buoy_s_02` | circle | center=`(318.0, 28.0)`, radius=`6.0` |
| `rock_circle_s_03` | circle | center=`(75.0, 96.0)`, radius=`5.0` |
| `rock_ellipse_s_04` | ellipse | center=`(122.0, 105.0)`, radii=`(8.0, 3.5)`, yaw=`40 deg` |
| `l_shape_island_w_01` | polygon/L-shape | vertices=`(45,155) -> (105,155) -> (105,183) -> (78,183) -> (78,250) -> (45,250)` |
| `u_shape_island_wc_01` | polygon/U-shape | vertices=`(135,140) -> (215,140) -> (215,166) -> (174,166) -> (174,214) -> (215,214) -> (215,240) -> (135,240) -> (135,214) -> (151,214) -> (151,166) -> (135,166)` |
| `s_curve_barrier_c_01` | polygon/S-curve | moderate curved band from approximately `(258,120)` to `(276,274)`, represented by a 14-vertex simple polygon |
| `rotated_barrier_nw_01` | rectangle | center=`(88.0, 294.0)`, size=`58.0 x 8.0`, yaw=`25 deg` |
| `triangle_shoal_e_01` | polygon/triangle | vertices=`(332,240) -> (382,258) -> (346,292)` |
| `ellipse_reef_nw_01` | ellipse | center=`(70.0, 330.0)`, radii=`(34.0, 12.0)`, yaw=`-10 deg` |
| `trapezoid_island_n_01` | polygon/trapezoid | vertices=`(112,318) -> (176,306) -> (196,350) -> (126,370)` |
| `irregular_island_ne_01` | polygon/irregular | vertices=`(300,318) -> (328,300) -> (370,312) -> (382,350) -> (340,378) -> (302,360)` |
| `circle_islet_ne_01` | circle | center=`(372.0, 386.0)`, radius=`6.0` |
| `ellipse_reef_e_02` | ellipse | center=`(336.0, 124.0)`, radii=`(14.0, 5.5)`, yaw=`-35 deg` |
| `small_triangle_w_02` | polygon/triangle | vertices=`(28,286) -> (52,276) -> (46,312)` |
| `rock_circle_cluster_01` | circle | center=`(248.0, 40.0)`, radius=`4.5` |
| `rock_circle_cluster_02` | circle | center=`(264.0, 52.0)`, radius=`3.5` |
| `rock_ellipse_cluster_03` | ellipse | center=`(244.0, 92.0)`, radii=`(7.0, 3.0)`, yaw=`62 deg` |
| `rock_polygon_cluster_04` | polygon/irregular | vertices=`(118,266) -> (130,258) -> (146,270) -> (140,286) -> (122,284)` |
| `rect_platform_n_02` | rectangle | center=`(250.0, 384.0)`, size=`34.0 x 12.0`, yaw=`5 deg` |

## S 型弧形障碍

`s_curve_barrier_c_01` 使用当前 loader 支持的 `polygon` 表达，不引入新的障碍类型。它近似一条加宽的 S 形弯曲带，中心线大致经过：

```text
(258,120) -> (286,144) -> (294,170) -> (278,196)
-> (254,222) -> (250,250) -> (276,274)
```

该障碍用于制造中等尺度弯曲边界和局部凹凸自由空间，但避免过长、过大的弧度切割整张地图。

## 设计说明

- 障碍类型全部来自当前地图加载器支持的类型：`rectangle`、`ellipse`、`circle`、`polygon`。
- 地图包含规则障碍、旋转障碍、曲边障碍、凹多边形障碍、不规则多边形障碍和长弧形障碍。
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
outputs/static_obstacle_map_400x400_mixed_obstacles_usv8_footprint4x2_rmin2/
```
