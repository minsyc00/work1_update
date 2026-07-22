# static_obstacle_map_20x20_two_obstacles

## 用途

该地图用于快速验证多无人艇静态障碍全覆盖路径规划算法。地图只包含任务区域和静态障碍物，不包含无人艇初始状态、路径结果、覆盖条带或仿真轨迹。

## 任务区域

- 区域类型：矩形水域。
- 坐标范围：`[0, 20] x [0, 20]`。
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
| `rect_dock_01` | rectangle | center=`(7.0, 7.0)`, size=`2.0 x 2.5`, yaw=`0 deg` |
| `elliptic_reef_01` | ellipse | center=`(14.0, 13.0)`, radii=`(1.4, 0.8)`, yaw=`-25 deg` |

## 不包含内容

- 不包含无人艇数量或初始位姿。
- 不包含动态障碍物。
- 不包含规划路径。
- 不包含轨迹参考或闭环仿真结果。

## 本次推荐实验输出目录

```text
D:\code\work1_update\outputs\static_obstacle_map_20x20_two_obstacles_usv2_footprint4x2_rmin2\
```

所有算法运行结果应保存到该输出目录，而不是直接保存到 `outputs` 根目录。
