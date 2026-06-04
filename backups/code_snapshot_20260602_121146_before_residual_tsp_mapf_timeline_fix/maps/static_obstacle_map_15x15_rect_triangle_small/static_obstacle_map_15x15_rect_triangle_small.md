# static_obstacle_map_15x15_rect_triangle_small

## 用途

该地图用于验证静态障碍物场景下的无人艇覆盖路径规划。地图只包含任务区域、一个小矩形障碍物和一个小三角形障碍物，不包含无人艇初始状态、规划路径、覆盖条带或仿真轨迹。

## 任务区域

- 区域类型：矩形水域。
- 坐标范围：`[0, 15] x [0, 15]`。
- 单位：`m`。
- 原点：左下角 `(0, 0)`。

## 覆盖与运动约束

- 覆盖足迹：矩形。
- 覆盖足迹长度：`length_lf = 4.0 m`。
- 覆盖足迹宽度：`width_wf = 2.0 m`。
- 无人艇模型：`3-DOF`。
- 最小转弯半径：`Rmin = 2.0 m`。
- 推荐重叠率：`rho = 0.2`。
- 推荐安全距离：`d_safe = 0.5 m`。

## 静态障碍物

| ID | 类型 | 参数 |
| --- | --- | --- |
| `small_rect_01` | rectangle | center=`(5.0, 5.5)`, size=`1.4 x 1.2`, yaw=`0 deg` |
| `small_triangle_01` | polygon/triangle | vertices=`(10.0, 9.5)`, `(11.4, 9.2)`, `(10.7, 10.7)` |

## 不包含内容

- 不包含无人艇数量或初始位姿。
- 不包含动态障碍物。
- 不包含规划路径。
- 不包含轨迹参考或闭环仿真结果。

## 推荐实验输出目录

```text
D:\code\work1_update\outputs\static_obstacle_map_15x15_rect_triangle_small_usv2_footprint4x2_rmin2\
```

所有算法运行结果应保存到对应的 `outputs` 子目录，而不是直接保存到 `outputs` 根目录。
