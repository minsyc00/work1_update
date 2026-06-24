from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, wait
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .nmpc import CasadiNMPCController, NMPCResult
from .schema import ControlInput, PlannerConfig, State3DOF, TrajectorySample


@dataclass
class NMPCSolveRequest:
    agent_id: int
    state: State3DOF
    previous_control: ControlInput
    ref_window: List[TrajectorySample]
    preferred_velocities: np.ndarray
    neighbor_predictions: List[List[TrajectorySample]]
    obstacle_predictions: List[List[Tuple[float, float, float]]]
    mismatch: np.ndarray
    safe_distance: float


@dataclass
class NMPCBackendSolveResult:
    agent_id: int
    result: NMPCResult | None
    solve_time_ms: float
    timed_out: bool = False
    error: str = ""


class ProcessCasadiNMPCBackend:
    def __init__(
        self,
        config: PlannerConfig,
        *,
        horizon_steps: int,
        dt: float,
        max_neighbors: int,
        max_obstacles: int,
        model_params: Dict[str, float],
        max_workers: int,
    ) -> None:
        self.config = config
        self.horizon_steps = horizon_steps
        self.dt = dt
        self.max_neighbors = max_neighbors
        self.max_obstacles = max_obstacles
        self.model_params = dict(model_params)
        self.max_workers = max(1, int(max_workers))
        self._executor: ProcessPoolExecutor | None = None
        self.worker_restart_count = 0
        self.hard_timeout_count = 0

    def solve_many(self, requests: Sequence[NMPCSolveRequest], timeout_ms: float) -> Dict[int, NMPCBackendSolveResult]:
        if not requests:
            return {}
        executor = self._ensure_executor()
        future_to_agent = {
            executor.submit(_solve_request_in_worker, request): request.agent_id
            for request in requests
        }
        timeout_s = max(float(timeout_ms), 1.0) / 1000.0
        done, not_done = wait(list(future_to_agent), timeout=timeout_s)
        results: Dict[int, NMPCBackendSolveResult] = {}
        for future in done:
            agent_id = future_to_agent[future]
            try:
                results[agent_id] = future.result()
            except BaseException as exc:
                results[agent_id] = NMPCBackendSolveResult(
                    agent_id=agent_id,
                    result=None,
                    solve_time_ms=timeout_ms,
                    timed_out=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
        if not_done:
            self.hard_timeout_count += len(not_done)
            for future in not_done:
                future.cancel()
                agent_id = future_to_agent[future]
                results[agent_id] = NMPCBackendSolveResult(
                    agent_id=agent_id,
                    result=None,
                    solve_time_ms=float(timeout_ms),
                    timed_out=True,
                    error="process_hard_timeout",
                )
            self._restart_executor(terminate_workers=True)
        return results

    def close(self, *, terminate_workers: bool = False) -> None:
        self._shutdown_executor(terminate_workers=terminate_workers)

    def _shutdown_executor(self, *, terminate_workers: bool) -> None:
        if self._executor is not None:
            if terminate_workers:
                # ProcessPoolExecutor cannot hard-cancel a running IPOPT solve via
                # Future.cancel(); terminate the worker processes before restart.
                for process in list((getattr(self._executor, "_processes", None) or {}).values()):
                    try:
                        if process.is_alive():
                            process.terminate()
                    except BaseException:
                        pass
                for process in list((getattr(self._executor, "_processes", None) or {}).values()):
                    try:
                        process.join(timeout=1.0)
                        if process.is_alive() and hasattr(process, "kill"):
                            process.kill()
                            process.join(timeout=1.0)
                        if hasattr(process, "close"):
                            process.close()
                    except BaseException:
                        pass
            self._executor.shutdown(wait=terminate_workers, cancel_futures=True)
            self._executor = None

    def _ensure_executor(self) -> ProcessPoolExecutor:
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=self.max_workers,
                initializer=_init_worker,
                initargs=(
                    self.config,
                    self.horizon_steps,
                    self.dt,
                    self.max_neighbors,
                    self.max_obstacles,
                    self.model_params,
                ),
            )
        return self._executor

    def _restart_executor(self, *, terminate_workers: bool = False) -> None:
        self.worker_restart_count += 1
        self._shutdown_executor(terminate_workers=terminate_workers)


_WORKER_CONFIG: PlannerConfig | None = None
_WORKER_HORIZON_STEPS = 0
_WORKER_DT = 0.0
_WORKER_MAX_NEIGHBORS = 1
_WORKER_MAX_OBSTACLES = 1
_WORKER_MODEL_PARAMS: Dict[str, float] = {}
_WORKER_CONTROLLERS: Dict[int, CasadiNMPCController] = {}


def _init_worker(
    config: PlannerConfig,
    horizon_steps: int,
    dt: float,
    max_neighbors: int,
    max_obstacles: int,
    model_params: Dict[str, float],
) -> None:
    global _WORKER_CONFIG
    global _WORKER_HORIZON_STEPS
    global _WORKER_DT
    global _WORKER_MAX_NEIGHBORS
    global _WORKER_MAX_OBSTACLES
    global _WORKER_MODEL_PARAMS
    global _WORKER_CONTROLLERS
    _WORKER_CONFIG = config
    _WORKER_HORIZON_STEPS = horizon_steps
    _WORKER_DT = dt
    _WORKER_MAX_NEIGHBORS = max_neighbors
    _WORKER_MAX_OBSTACLES = max_obstacles
    _WORKER_MODEL_PARAMS = dict(model_params)
    _WORKER_CONTROLLERS = {}


def _solve_request_in_worker(request: NMPCSolveRequest) -> NMPCBackendSolveResult:
    if _WORKER_CONFIG is None:
        raise RuntimeError("NMPC process worker is not initialized")
    controller = _WORKER_CONTROLLERS.get(request.agent_id)
    if controller is None:
        controller = CasadiNMPCController(
            config=_WORKER_CONFIG,
            horizon_steps=_WORKER_HORIZON_STEPS,
            dt=_WORKER_DT,
            max_neighbors=_WORKER_MAX_NEIGHBORS,
            max_obstacles=_WORKER_MAX_OBSTACLES,
            mass_u=_WORKER_MODEL_PARAMS["mass_u"],
            mass_v=_WORKER_MODEL_PARAMS["mass_v"],
            mass_r=_WORKER_MODEL_PARAMS["mass_r"],
            damp_u=_WORKER_MODEL_PARAMS["damp_u"],
            damp_v=_WORKER_MODEL_PARAMS["damp_v"],
            damp_r=_WORKER_MODEL_PARAMS["damp_r"],
            cross_coupling=_WORKER_MODEL_PARAMS["cross_coupling"],
        )
        _WORKER_CONTROLLERS[request.agent_id] = controller
    started = time.perf_counter()
    result = controller.solve(
        state=request.state,
        previous_control=request.previous_control,
        ref_window=request.ref_window,
        preferred_velocities=request.preferred_velocities,
        neighbor_predictions=request.neighbor_predictions,
        obstacle_predictions=request.obstacle_predictions,
        mismatch=request.mismatch,
        safe_distance=request.safe_distance,
    )
    return NMPCBackendSolveResult(
        agent_id=request.agent_id,
        result=result,
        solve_time_ms=(time.perf_counter() - started) * 1000.0,
    )
