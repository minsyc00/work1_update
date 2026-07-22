# static_obstacle_map_200x200_mixed_obstacles

## 用途

该地图是 `200m x 200m` 多形态静态障碍全覆盖路径规划实验环境资产。文件只描述任务区域和静态障碍物，不包含无人艇初始状态、覆盖路径、轨迹参考或仿真结果。

本版本已按当前区域分解、倾斜往复扫描、区域间连接和 residual 补扫算法做适配：障碍物保持圆形、椭圆、凸多边形、浅凹多边形和 S 型障碍等多形态，但数量控制在 8 个代表性障碍，并在障碍之间保留宽通道，避免深凹口、密集小礁石和窄死胡同导致区域分解碎片化或连接验证长时间搜索。

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
- 推荐无人艇数量：`4`。
- 推荐重叠率：`rho = 0.2`。
- 推荐安全距离：`d_safe = 1.0 m`。

## 静态障碍物

| ID | 类型 | 参数 |
| --- | --- | --- |
| `rect_platform_sw_01` | rectangle | center=`(32.0, 34.0)`, size=`16.0 x 12.0`, yaw=`0 deg` |
| `rotated_dock_sw_01` | rectangle | center=`(72.0, 38.0)`, size=`26.0 x 5.0`, yaw=`-20 deg` |
| `ellipse_reef_s_01` | ellipse | center=`(116.0, 36.0)`, radii=`(11.0, 4.5)`, yaw=`18 deg` |
| `circle_reef_e_01` | circle | center=`(164.0, 86.0)`, radius=`7.0` |
| `shallow_concave_island_e_01` | polygon/shallow-concave | vertices=`(150,112) -> (184,112) -> (190,130) -> (170,126) -> (160,136) -> (148,128)` |
| `s_curve_lower_arc_01` | ellipse/S-chain | center=`(58.0, 146.0)`, radii=`(14.0, 3.5)`, yaw=`28 deg` |
| `s_curve_upper_arc_02` | ellipse/S-chain | center=`(66.0, 174.0)`, radii=`(14.0, 3.5)`, yaw=`-28 deg` |
| `north_convex_island_01` | polygon/convex | vertices=`(104,158) -> (144,152) -> (176,172) -> (156,196) -> (116,190) -> (96,170)` |

## S 型障碍说明

`s_curve_lower_arc_01` 和 `s_curve_upper_arc_02` 使用两段不重叠椭圆表达短 S 型障碍链。两段之间保留通道，不形成连续长屏障；主要用于测试倾斜边界和弯曲障碍附近的区域分解。

## 适配当前算法的设计说明

- 保留一个浅凹多边形 `shallow_concave_island_e_01`，凹口较宽且凹入较浅，不形成 U 形死胡同。
- S 型障碍使用两段短椭圆链表达，不使用长凹多边形切开地图。
- 障碍物之间保留足够通道宽度，避免膨胀障碍后形成极窄连通缝隙。
- 西侧、中部、北部和东北侧都保留较大的连续自由水域，便于大块凸区域分解和简单往复扫描。
- 小礁石簇已移除，避免全局障碍坐标切分产生过多小区域。
- 障碍类型全部来自当前地图加载器支持的类型：`rectangle`、`ellipse`、`circle`、`polygon`。

## 不包含内容

- 不包含无人艇初始状态。
- 不包含动态障碍物。
- 不包含覆盖路径。
- 不包含轨迹参考。
- 不包含仿真结果。

## 推荐输出目录

后续如果运行实验，推荐输出目录为：

```text
outputs/static_obstacle_map_200x200_mixed_obstacles_usv4_footprint4x2_rmin2/
```
