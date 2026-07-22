# P1 级问题详细修复计划：安全保证、可扩展 MAPF 与实时 NMPC 工程化

## Summary

本计划针对 `plans/usv_bottleneck_report_2026-06-22.md` 中的 P1 级严重问题，目标是在 P0 修复后的基础上，把当前多 USV 3-DOF 覆盖系统从“默认可运行”推进到“可扩展、可审计、可部署原型”。

P1 修复范围固定为三条主线：

1. 形式化碰撞避免：把当前软约束安全项升级为可配置硬 CBF / CBF-QP 安全过滤，保证艇间、边界、障碍距离不靠 slack 碰运气。
2. CBS-MAPF 可扩展性：给 CBS / resource-window 调度加入搜索预算、冲突索引、退化策略和可诊断报告，避免 N 增大后指数爆炸或长时间卡死。
3. 真正并行与硬超时 NMPC：补齐 process backend / worker-local controller / hard timeout，解决 Python GIL 与 CasADi thread 不安全问题；同时保留 Acados 迁移接口作为下一阶段高速求解器。

当前审计状态：

- P0 覆盖栅格局部更新已完成。
- P0 3-DOF RK4 积分与 NMPC RK4 预测已完成。
- P0 默认 `hybrid_nmpc` 已能缓解实时性问题。
- 但真实 CasADi `thread` backend 不安全，当前已自动降级为 `serial_casadi_thread_disabled`；P1 需要用 process backend 或 Acados 正式解决。

## P1-1: 形式化碰撞避免与安全过滤

### 当前问题

当前 `src/usv_swarm/nmpc.py` 中艇间、障碍、边界安全约束使用 slack：

- `s_nei`
- `s_obs`
- `s_bound`

约束形式类似：

```text
h_cur + slack >= 0
h_next - (1 - gamma) * h_cur + slack >= 0
objective += w_soft * slack^2
```

这意味着安全约束可以被违反，只是通过代价惩罚；在高密度、多艇交会或 NMPC 超时 fallback 时，缺乏形式化安全保证。

### 修复目标

- 新增安全模式：`soft_cbf`、`hard_cbf`、`cbf_qp_filter`、`hybrid_cbf_qp`。
- 默认保持工程可行的 `hybrid_cbf_qp`：正常由 tracker/NMPC 输出控制，最后统一过 CBF-QP 安全过滤。
- NMPC 中支持硬 CBF 约束：可配置是否允许 slack。
- tracker fallback / timeout fallback 后也必须经过安全过滤。
- 安全报告能区分：安全约束真实满足、CBF-QP 修正、safe-hold 介入、不可恢复风险。

### 实现步骤

1. 扩展配置。
   - 文件：`src/usv_swarm/schema.py`
   - 新增字段：
     - `safety_filter_mode: str = "hybrid_cbf_qp"`
     - `cbf_alpha: float = 0.8`
     - `cbf_allow_slack: bool = False`
     - `cbf_slack_weight: float = 1e4`
     - `cbf_qp_max_iter: int = 30`
     - `cbf_qp_timeout_ms: float = 5.0`
     - `safety_min_margin_epsilon: float = 1e-3`

2. 新增 CBF 安全过滤模块。
   - 新文件：`src/usv_swarm/safety_filter.py`
   - 核心接口：
     - `filter_control_cbf_qp(state, nominal_control, neighbors, obstacles, bounds, config) -> SafetyFilterResult`
   - 输出内容：
     - `control`
     - `feasible`
     - `active_constraints`
     - `min_predicted_margin`
     - `slack_used`
     - `solve_time_ms`
     - `fallback_reason`

3. 将 CBF-QP 接入控制链路。
   - 文件：`src/usv_swarm/control.py`
   - 在 `_finalize_control_context(...)` 中，对 `tracker`、`hybrid_nmpc`、`full_nmpc` 和 `safe_hold` 输出统一调用 safety filter。
   - 若 QP 可行，替换控制输入并记录 `safety_status.mode += "+cbf_filtered"`。
   - 若 QP 不可行，进入 `safe_hold` 或低速停车，并记录 `cbf_filter_failed`。

4. NMPC 支持硬安全约束。
   - 文件：`src/usv_swarm/nmpc.py`
   - 当 `cbf_allow_slack=False` 时：
     - 艇间 `h >= 0` 不加 `s_nei`。
     - 障碍 `h >= 0` 不加 `s_obs`。
     - 边界 `h >= 0` 不加 `s_bound`。
   - 当 `cbf_allow_slack=True` 时保留 slack，但在报告中统计 `max_slack` 和 `slack_violation_count`。

5. 增加安全性能监控。
   - 文件：`src/usv_swarm/simulation.py`
   - runtime profile 新增：
     - `cbf_filter_called_count`
     - `cbf_filter_failed_count`
     - `cbf_filter_avg_time_ms`
     - `safety_min_distance`
     - `boundary_violation_count`
     - `obstacle_violation_count`
     - `cbf_slack_used_count`

### 验收标准

- 会遇、交叉、追越、窄通道入口、动态障碍插入 5 类场景中：
  - 艇间最小距离 `>= d_safe - 1e-3`
  - 边界违规计数为 `0`
  - 障碍违规计数为 `0`
- NMPC timeout 时仍由 CBF-QP 或 safe-hold 保证安全。
- `cbf_filter_avg_time_ms <= 5ms`，否则必须自动降级到解析 safe-hold。
- 报告中明确给出是否使用 slack；默认验收要求 `cbf_slack_used_count == 0`。

### 测试计划

- 单元测试：两艇相向接近时，CBF-QP 会降低速度或转向，预测 `h_next >= 0`。
- 单元测试：边界附近向外控制会被过滤为向内或停车。
- 单元测试：动态障碍插入时，过滤后控制不减少安全裕度。
- 集成测试：`fast_tracker`、`hybrid_nmpc`、`full_nmpc` 三种模式均经过安全过滤。
- 回归测试：`python -m unittest discover -s tests -v` 全部通过。

## P1-2: CBS-MAPF 与资源调度可扩展性

### 当前问题

当前 `src/usv_swarm/planning.py` 中 CBS 主循环基于 `while open_set`，缺少明确预算：

```text
while open_set:
    node = heapq.heappop(open_set)
    conflict = _find_first_conflict(node.reservations)
    ...
```

风险：

- 最坏情况下 CBS 指数爆炸。
- `_find_first_conflict(...)` 对所有 reservation 做朴素冲突扫描，复杂度随 agent 和资源窗口快速增长。
- 大图路径规划层已有 `apply_resource_window_schedule(...)`，但只是 lightweight hook，还不是带预算、可诊断、可退化的完整调度器。

### 修复目标

- CBS 必须有节点预算、冲突预算、wall-time 预算。
- 冲突检测从朴素全扫描升级为按 `resource_id` 分桶的 interval sweep。
- 失败时不静默返回 root，而是给出 `budget_exhausted`、未解决冲突列表和退化调度结果。
- 对大图论文式路径规划，优先使用 resource-window scheduler + conflict graph；对小图保留 CBS。
- 支持 bounded-suboptimal / prioritized fallback，避免卡死。

### 实现步骤

1. 扩展 MAPF 配置。
   - 文件：`src/usv_swarm/schema.py`
   - 新增字段：
     - `mapf_solver: str = "auto"`
     - `mapf_max_expanded_nodes: int = 5000`
     - `mapf_max_conflicts: int = 2000`
     - `mapf_max_wall_time_ms: float = 2000.0`
     - `mapf_suboptimality_bound: float = 1.2`
     - `mapf_fallback: str = "prioritized_resource_windows"`

2. 优化冲突检测。
   - 文件：`src/usv_swarm/planning.py`
   - 新增：
     - `_build_resource_interval_index(reservations)`
     - `_find_first_conflict_indexed(index)`
   - 资源类型：
     - vertex
     - directed edge
     - reverse edge
     - turn pocket
     - narrow corridor
     - coverage strip
   - 冲突检测按资源分桶排序，只比较相邻重叠窗口。

3. 给 CBS 加预算与诊断。
   - 文件：`src/usv_swarm/planning.py`
   - `solve_cbs_mapf(...)` 新增返回字段：
     - `solver_status`
     - `expanded_nodes`
     - `open_set_peak`
     - `conflict_checks`
     - `budget_exhausted`
     - `unresolved_conflicts`
     - `fallback_used`

4. 实现 prioritized fallback。
   - 文件：`src/usv_swarm/planning.py` 或新文件 `src/usv_swarm/mapf.py`
   - 按 agent 优先级逐个插入时间窗：
     - 高负载 agent 优先。
     - 已在窄通道中的 agent 优先。
     - 残差补扫 agent 后置。
   - 如果发生冲突，后续 agent 延迟或重排局部连接。

5. 统一路径规划层资源调度。
   - 文件：
     - `src/usv_swarm/path_planning/scheduling.py`
     - `src/usv_swarm/path_planning/resources.py`
     - `src/usv_swarm/path_planning/pipeline.py`
     - `src/usv_swarm/path_planning/paper_style_experiment.py`
   - 将当前 `apply_resource_window_schedule(...)` 升级为：
     - 可返回 unresolved conflicts。
     - 可输出 schedule diagnostics。
     - 可应用相同预算配置。

### 验收标准

- `N=2/4/8` 多艇路径调度不会无界运行。
- 对复杂交叉路径，CBS 超预算时必须返回 `fallback_used=True`，并输出剩余冲突。
- 对窄通道、入口、turn pocket、coverage strip，最终 `true_time_conflict_count == 0`，若不能为 0，报告必须列出具体 resource_id。
- `mapf_max_wall_time_ms=2000` 时，调度阶段不会超过预算 20% 以上。

### 测试计划

- 单元测试：indexed conflict detector 与旧 detector 在小场景结果一致。
- 单元测试：构造大量不相交资源，indexed detector 比旧 detector 比较次数明显减少。
- 单元测试：设置 `mapf_max_expanded_nodes=1` 时触发 fallback，结果包含诊断。
- 集成测试：8 艘艇交叉覆盖路径在预算内完成调度。
- 可视化测试：`10_shared_resource_timeline.png` 能标出 unresolved conflicts 与 fallback schedule。

## P1-3: 真正并行 NMPC 与硬超时

### 当前问题

当前已实现：

- `fast_tracker`
- `hybrid_nmpc`
- `full_nmpc`
- `thread` backend 的测试桩并行

但真实 CasADi `Opti` 在线程中不可靠，当前运行时已自动降级为：

```text
nmpc_parallel_backend_effective = "serial_casadi_thread_disabled"
```

此外，`nmpc_max_wall_time_ms` 当前主要是求解返回后的判定，不是真正能中断 IPOPT 的硬 timeout。

### 修复目标

- 新增 `process` backend，使用 worker-local CasADi controller。
- 每个 worker 内部持有自己的 `CasadiNMPCController`，避免 pickle `Opti` 对象。
- 主进程只传可序列化输入。
- `Future.result(timeout=...)` 超时后，丢弃该 worker 结果，并可重启 worker。
- fallback 不阻塞主循环。
- 为 Acados 迁移预留统一 `NMPCBackend` 接口。

### 实现步骤

1. 抽象 NMPC backend。
   - 新文件：`src/usv_swarm/nmpc_backend.py`
   - 接口：
     - `solve(agent_id, request) -> NMPCResult`
     - `close()`
     - `backend_status()`
   - 实现：
     - `SerialCasadiBackend`
     - `ProcessCasadiBackend`
     - `AcadosBackend` 占位

2. 定义可序列化请求。
   - 新 dataclass：
     - `NMPCSolveRequest`
   - 字段：
     - `agent_id`
     - `state`
     - `previous_control`
     - `ref_window`
     - `preferred_velocities`
     - `neighbor_predictions`
     - `obstacle_predictions`
     - `mismatch`
     - `safe_distance`
     - `config_snapshot`

3. 实现 worker-local controller。
   - worker 初始化时构造 `CasadiNMPCController`。
   - 每个 worker 绑定一个 agent，保留 warm start。
   - 不在进程间传递 `Opti`。

4. 实现硬 timeout。
   - 使用 `concurrent.futures.ProcessPoolExecutor` 或长期存活 worker process。
   - 主进程：
     - `future.result(timeout=nmpc_max_wall_time_ms / 1000)`
     - timeout 后立即 fallback 到 tracker 或上一次有效控制。
     - 标记 worker stale，必要时重启。

5. 接入 `SwarmRuntime.control_steps(...)`。
   - `serial`：当前行为。
   - `thread`：仅允许非 CasADi 测试桩或未来线程安全 backend。
   - `process`：真实多 agent NMPC 推荐模式。
   - profile 新增：
     - `nmpc_parallel_backend_effective`
     - `nmpc_worker_restart_count`
     - `nmpc_hard_timeout_count`
     - `nmpc_process_queue_time_ms`

6. Acados 迁移接口。
   - 新增 `AcadosBackend` 空壳实现，若依赖未安装则返回清晰错误。
   - 配置：
     - `nmpc_solver_backend: str = "casadi"`
     - 可选值：`casadi`、`acados`
   - 后续迁移时只替换 backend，不改 control runtime。

### 验收标准

- `N=5, local_control_hz=5, backend=process`：
  - 平均 real-time factor `>= 0.8`
  - 单 step 不因某个 IPOPT 卡住超过 `nmpc_max_wall_time_ms + 30ms`
  - timeout 后 fallback 正常，仿真不中断
- `backend=thread` + CasADi：
  - 不崩溃
  - 明确降级为 `serial_casadi_thread_disabled`
- `backend=process` worker 异常：
  - 主进程继续运行
  - worker restart 计数增加
- 不传 backend 参数时，默认行为保持 `hybrid_nmpc + serial`。

### 测试计划

- 单元测试：`NMPCSolveRequest` 可 pickle。
- 单元测试：Process backend worker-local controller 能返回 NMPCResult。
- 单元测试：构造 slow worker，验证 hard timeout 后 fallback 且主循环不阻塞。
- 集成测试：N=5, 5Hz, 2s 仿真，输出 profiler。
- 回归测试：`python -m unittest discover -s tests -v` 全部通过。

## P1-4: Acados 迁移路线

### 目的

Acados 不是第一天必须完成的改动，但它是从“可运行原型”走向“实时部署”的关键路线。P1 阶段先做接口隔离和最小可运行 demo，避免一次性重写整个 NMPC。

### 实施顺序

1. 增加 optional dependency。
   - 文件：`pyproject.toml`
   - extras：
     - `realtime = ["acados_template", ...]`

2. 新增 Acados 模型生成脚本。
   - 新文件：`tools/generate_acados_usv3dof_solver.py`
   - 输出目录：
     - `build/acados/usv3dof/`

3. 实现最小 Acados backend。
   - 文件：`src/usv_swarm/nmpc_backend.py`
   - 与 CasADi backend 使用相同 `NMPCSolveRequest`。

4. 对照测试。
   - 同一参考轨迹：
     - CasADi objective
     - Acados objective
     - solve time
     - min safety margin

### Acados 验收标准

- 单艇 horizon=10 求解时间目标 `< 10ms`。
- N=5 process 或 native backend 平均 real-time factor `>= 1.0`。
- 与 CasADi 相同场景下轨迹误差不超过 `1.25x`。
- 若 Acados 未安装，普通测试不失败，只跳过 `requires_acados` 测试。

## P1-5: 监控、报告与实验命令

### 新增报告字段

控制层：

- `safety_filter_mode`
- `cbf_filter_called_count`
- `cbf_filter_failed_count`
- `cbf_slack_used_count`
- `safety_min_distance`
- `boundary_violation_count`
- `obstacle_violation_count`
- `nmpc_parallel_backend`
- `nmpc_parallel_backend_effective`
- `nmpc_hard_timeout_count`
- `nmpc_worker_restart_count`

MAPF 层：

- `mapf_solver`
- `mapf_solver_status`
- `mapf_expanded_nodes`
- `mapf_open_set_peak`
- `mapf_conflict_checks`
- `mapf_budget_exhausted`
- `mapf_fallback_used`
- `unresolved_conflict_count`

### 推荐命令

基础回归：

```powershell
D:\anaconda3\envs\pytorch_gpu\python.exe -m unittest discover -s tests -v
```

P1 安全过滤 smoke：

```powershell
D:\anaconda3\envs\pytorch_gpu\python.exe examples\closed_loop_simulation.py --control-mode hybrid_nmpc --nmpc-update-interval 5 --nmpc-max-wall-time-ms 80
```

P1 MAPF 压力测试：

```powershell
D:\anaconda3\envs\pytorch_gpu\python.exe examples\run_paper_style_region_tsp_experiment.py --map maps\static_obstacle_map_50x50_simple\static_obstacle_map_50x50_simple.json --usv-count 8 --rmin 1.0 --monitor-stages --no-render
```

P1 process backend 测试：

```powershell
D:\anaconda3\envs\pytorch_gpu\python.exe examples\closed_loop_simulation.py --control-mode hybrid_nmpc --nmpc-update-interval 1 --nmpc-max-wall-time-ms 80 --nmpc-parallel-backend process
```

## 分阶段排期

### Week 1: 安全过滤与 MAPF 预算

- 完成 `safety_filter.py` 最小 CBF-QP。
- 将 tracker/NMPC 输出统一过 filter。
- CBS 增加预算与诊断字段。
- 完成关键单测。

### Week 2: MAPF 索引与 fallback

- 完成 resource interval index。
- 完成 prioritized fallback。
- 升级 `10_shared_resource_timeline.png` 诊断。
- 跑 N=8 资源冲突压力测试。

### Week 3: process NMPC backend

- 完成 `nmpc_backend.py`。
- 实现 worker-local CasADi controller。
- 实现 hard timeout 与 worker restart。
- 完成 N=5/N=8 控制性能测试。

### Week 4: Acados 接口与对照实验

- 完成 Acados optional backend scaffold。
- 支持依赖缺失时跳过测试。
- 输出 CasADi vs Acados solve-time 对照报告。

## 总体验收门槛

P1 修复完成必须同时满足：

- 完整单元测试通过。
- `N=5, 5Hz` 闭环仿真平均 real-time factor `>= 0.8`。
- `N=8` MAPF/resource 调度不超预算卡死。
- 安全违规计数为 `0` 或报告明确进入 safe-hold 且没有越界/碰撞。
- 真实 CasADi 不再在线程中崩溃；用户请求 `thread` 时有明确降级，用户请求 `process` 时使用真正进程隔离。
- 所有失败都必须有诊断字段，不允许静默回退成看似成功的结果。

## 风险与降级策略

- CBF-QP 可能在高密度场景不可行：降级到低速停车 + resource freeze。
- Process backend 会增加序列化开销：保留 `hybrid_nmpc` 低频触发，避免每步大量 IPC。
- Acados 安装成本高：作为 optional backend，不阻塞普通 CI。
- MAPF fallback 可能不是全局最优：报告 `fallback_used=True`，并用 bounded-suboptimal 指标说明。

