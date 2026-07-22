# USV P1 修复完成状态

## Summary
依据 `plans/usv_bottleneck_report_2026-06-22.md` 的 P1 清单，当前 P1 #4-#7 均已有代码路径、配置入口、测试覆盖和运行时诊断。Acados 采用可选后端策略：默认 `auto` 会尝试 Acados，当前环境不可用时回退到 CasADi/IPOPT，并在 profile 中记录原因。

## P1 Items
- `#4 Acados 迁移`：新增 `nmpc_solver_backend=auto|casadi|acados` 与可选 Acados 后端入口；当前不强制安装 Acados，`auto` 模式可回退 CasADi。
- `#5 并行 NMPC 求解`：已支持 `process` 后端，worker-local controller，硬超时后终止并重建 worker。
- `#6 CBF 硬约束碰撞避免`：NMPC 中支持无 slack CBF 约束，runtime 侧有 CBF-QP safety filter。
- `#7 CBS-MAPF 迭代上限`：已加入搜索预算、冲突索引和 prioritized resource-window fallback。

## Runtime Diagnostics
- `nmpc_solver_backend_requested`
- `nmpc_solver_backend_effective`
- `acados_available`
- `acados_fallback_reason`
- `nmpc_parallel_backend_effective`
- `nmpc_hard_timeout_count`
- `nmpc_worker_restart_count`

## Notes
当前实现只针对 3-DOF 控制层。Acados 是可选依赖；若需要真实 Acados 求解，需要在本地安装并配置 `acados_template` 与 acados native runtime。
