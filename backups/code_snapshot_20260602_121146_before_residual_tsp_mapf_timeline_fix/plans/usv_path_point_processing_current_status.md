# 路径点处理现状说明

## 当前主链路

当前路径规划层的路径点处理流程为：

```text
地图/障碍物
-> 障碍物膨胀
-> 自由空间 cell 分解
-> CoveragePass
-> RegionCoveragePattern
-> RegionGraph
-> 负载均衡
-> 单艇 TSP-CPP
-> Dubins/Bezier 段
-> PathWaypoint
-> TrajectoryReference
```

## PathWaypoint

`PathWaypoint` 是路径规划层输出的基础路径点，字段为：

```text
x      # 全局坐标 x
y      # 全局坐标 y
psi    # 航向角
time   # 时间戳，可选
speed  # 参考速度，可选
```

这些路径点后续可转换为 `TrajectoryReference`，供已有 3-DOF/NMPC 跟踪层使用。

## PathSegmentSpec

路径点按段组织在 `PathSegmentSpec` 中，字段包括：

```text
kind           # cover / transit / turn / wait 等
waypoints      # PathWaypoint[]
curvature_max  # 当前段最大曲率
length         # 当前段长度
path_source    # straight / bezier / dubins_fallback / astar_dubins 等
metadata       # resource_id、dubins_modes、collision_free 等
```

其中覆盖段 `cover` 通常为直线，转弯段 `turn` 和区域间连接 `transit` 优先用 Dubins/Bezier 生成。

## 已支持能力

- 支持基础多艇区域分配。
- 支持每艘艇对自己的区域集合求解单艇 TSP-CPP。
- 支持每个区域选择候选覆盖模式、入口和出口。
- 支持 Dubins 最小转弯半径约束。
- 支持五次 Bezier 平滑，曲率不可行时回退 Dubins。
- 支持路径点时间参数化。
- 支持转换为 `SmoothedPath` 和 `TrajectoryReference`。

## 3-DOF 与转弯半径

当前高层路径规划使用 Dubins 曲线作为 3-DOF 可跟踪性的几何代理，保证：

```text
位置连续
航向连续
曲率受限
最小转弯半径受限
速度可时间参数化
```

曲率约束为：

```text
kappa <= 1 / Rmin + 1e-3
```

真实 3-DOF 动力学跟踪由已有 NMPC/控制层完成。

## 当前不足

- 静态障碍地图的自由空间分解仍偏保守。
- 障碍周围可能丢失部分可覆盖自由空间。
- 跨障碍区域连接还没有完整地把 A* 绕障通道转换为 Dubins/Bezier 子段。
- 新路径规划层还没有完全重新接入 MAPF/CBS 时间调度。
- 残差补扫目前具备检测和分配入口，但还不是完整二次规划闭环。

## 结论

当前系统是可运行工程版，已经具备基础多艇 TSP-CPP、Dubins/Bezier 路径点生成和 3-DOF 可跟踪轨迹输出能力。

但对于复杂静态障碍地图的论文级完整全覆盖任务，还需要继续增强自由空间分解、障碍裁剪覆盖、A* 绕障连接、MAPF/CBS 调度和残差补扫闭环。
