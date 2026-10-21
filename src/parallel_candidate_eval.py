"""Process-parallel candidate evaluation for the existing BKM semantics.

The tree/split loop and proposal commitment remain sequential. A session owns a
read-only NumPy memmap and a persistent spawn-based process pool for one fixed X;
multiple estimators/seeds can reuse it. Candidate and top-r child tasks are
submitted in stable order and collected in that same order.
"""
from __future__ import annotations

import multiprocessing as mp
import json
import os
import queue
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from src.bkm import BKM
from src.evaluation import cluster_sse, sse_for_indices
from src.split_rules import (
    SplitProposal,
    propose_split,
    savaresi_shape_index,
    split_gain,
)

_WORKER_X: np.ndarray | None = None
_WORKER_THREAD_LIMITER: Any = None
_WORKER_WARMUP_EVENT: Any = None
_WORKER_READY_QUEUE: Any = None


def _worker_init(memmap_path: str, warmup_event: Any, ready_queue: Any) -> None:
    """Open shared X read-only and cap BLAS/OpenMP in each spawned worker."""
    global _WORKER_X, _WORKER_THREAD_LIMITER, _WORKER_WARMUP_EVENT, _WORKER_READY_QUEUE
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    _WORKER_THREAD_LIMITER = threadpool_limits(limits=1)
    _WORKER_X = np.load(memmap_path, mmap_mode="r", allow_pickle=False)
    _WORKER_WARMUP_EVENT = warmup_event
    _WORKER_READY_QUEUE = ready_queue
    try:
        import psutil

        affinity = psutil.Process().cpu_affinity()
    except (ImportError, AttributeError, OSError):
        affinity = None
    _WORKER_READY_QUEUE.put(
        {
            "pid": os.getpid(),
            "threadpools": threadpool_info(),
            "cpu_affinity": affinity,
            "start_method": mp.get_start_method(),
            "input_is_memmap": isinstance(_WORKER_X, np.memmap),
            "input_writeable": bool(_WORKER_X.flags.writeable),
            "input_shape": list(_WORKER_X.shape),
            "thread_limit_environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
        }
    )


def _warmup_wait() -> int:
    """Block one startup task per process so session setup includes pool spawn."""
    if _WORKER_WARMUP_EVENT is not None:
        _WORKER_WARMUP_EVENT.wait()
    return os.getpid()


def _candidate_task(task: tuple[int, np.ndarray, int, int, bool]) -> tuple[Any, ...]:
    cid, members, n_init, seed, capture_timing = task
    if _WORKER_X is None:
        raise RuntimeError("candidate worker has no shared X memmap")
    started = perf_counter() if capture_timing else None
    try:
        proposal = propose_split(
            _WORKER_X,
            members,
            cluster_id=cid,
            n_init=n_init,
            random_state=seed,
        )
    except Exception as exc:
        if not capture_timing:
            raise
        return (
            cid, None, None, None, None, None,
            {
                "worker_id": os.getpid(),
                "worker_compute_seconds": perf_counter() - started,
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
    # Omit the redundant parent-members copy from process IPC; the main process
    # reconstructs it from the stable task list.
    return (
        cid,
        proposal.left_indices,
        proposal.right_indices,
        proposal.parent_sse,
        proposal.child_sse,
        proposal.gain,
        {
            "worker_id": os.getpid(),
            "worker_compute_seconds": perf_counter() - started,
            "success": True,
            "error": None,
        } if capture_timing else None,
    )


def _child_task(task: tuple[int, int, np.ndarray, int, int, bool]) -> tuple[Any, ...]:
    cid, side, members, n_init, seed, capture_timing = task
    if _WORKER_X is None:
        raise RuntimeError("candidate worker has no shared X memmap")
    if members.size < 2:
        return (
            cid, side, 0.0, 0,
            {
                "worker_id": os.getpid(),
                "worker_compute_seconds": 0.0,
                "success": True,
                "error": None,
            } if capture_timing else None,
        )
    started = perf_counter() if capture_timing else None
    try:
        gain = split_gain(_WORKER_X, members, n_init=n_init, random_state=seed)
    except Exception as exc:
        if not capture_timing:
            raise
        return (
            cid, side, None, 0,
            {
                "worker_id": os.getpid(),
                "worker_compute_seconds": perf_counter() - started,
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
    return (
        cid, side, float(gain), 1,
        {
            "worker_id": os.getpid(),
            "worker_compute_seconds": perf_counter() - started,
            "success": True,
            "error": None,
        } if capture_timing else None,
    )


class ParallelBKMSession:
    """Reusable worker/memmap session for repeated fits against one matrix.

    Session construction time (input staging, worker spawn, and warm-up) is
    exposed as ``setup_seconds``. Estimator core fit time remains in each
    estimator's ``parallel_timing_``. A session must be used as a context
    manager, its source X must remain unchanged for the session lifetime, and it
    must not be shared by concurrent parent threads.
    """

    def __init__(self, X: np.ndarray, n_jobs: int = 1, memmap_dir: str | None = None):
        if isinstance(n_jobs, bool) or int(n_jobs) != n_jobs or int(n_jobs) < 1:
            raise ValueError("n_jobs must be a positive integer")
        self.n_jobs = int(n_jobs)
        self._source_X = X
        self._input: np.ndarray | None = None
        self._memmap_dir = memmap_dir
        self.executor: ProcessPoolExecutor | None = None
        self._temporary_directory: tempfile.TemporaryDirectory[str] | None = None
        self._warmup_event: Any = None
        self.setup_seconds = 0.0
        self.worker_pids: tuple[int, ...] = ()
        self.worker_info: tuple[dict[str, Any], ...] = ()
        self.fit_calls = 0
        self._entered = False
        self._ever_entered = False

    def __enter__(self) -> "ParallelBKMSession":
        if self._ever_entered:
            raise RuntimeError("a ParallelBKMSession cannot be entered twice")
        self._ever_entered = True
        started = perf_counter()
        ready_queue = None
        try:
            # Include dtype conversion and contiguous staging in setup time;
            # preserve a private snapshot so later caller mutation is detected.
            self._input = np.array(
                self._source_X, dtype=np.float64, order="C", copy=True
            )
            if self._input.ndim != 2:
                raise ValueError("X must be a 2D array")
            self._temporary_directory = tempfile.TemporaryDirectory(
                prefix="adaptivebkm_parallel_", dir=self._memmap_dir
            )
            path = Path(self._temporary_directory.name) / "X.npy"
            writer = np.lib.format.open_memmap(
                path, mode="w+", dtype=np.float64, shape=self._input.shape
            )
            writer[:] = self._input
            writer.flush()
            del writer

            context = mp.get_context("spawn")
            self._warmup_event = context.Event()
            ready_queue = context.Queue()
            self.executor = ProcessPoolExecutor(
                max_workers=self.n_jobs,
                mp_context=context,
                initializer=_worker_init,
                initargs=(str(path), self._warmup_event, ready_queue),
            )
            startup = [self.executor.submit(_warmup_wait) for _ in range(self.n_jobs)]
            readiness = []
            try:
                for _ in range(self.n_jobs):
                    readiness.append(ready_queue.get(timeout=60.0))
            except queue.Empty as exc:
                raise RuntimeError("timed out while starting the worker pool") from exc
            self._warmup_event.set()
            startup_pids = {future.result() for future in startup}
            ready_pids = {int(info["pid"]) for info in readiness}
            self.worker_pids = tuple(sorted(ready_pids & startup_pids))
            if len(self.worker_pids) != self.n_jobs:
                # A barrier task per process should produce one PID each. Fail
                # explicitly because otherwise p would not be the requested p.
                raise RuntimeError(
                    f"worker warm-up reached {len(self.worker_pids)} processes; "
                    f"expected {self.n_jobs}"
                )
            self.worker_info = tuple(
                sorted(
                    (dict(info) for info in readiness if int(info["pid"]) in self.worker_pids),
                    key=lambda info: int(info["pid"]),
                )
            )
            ready_queue.close()
            ready_queue.join_thread()
            ready_queue = None
            self.setup_seconds = perf_counter() - started
            self._entered = True
            return self
        except BaseException:
            if self._warmup_event is not None:
                self._warmup_event.set()
            if self.executor is not None:
                self.executor.shutdown(wait=True, cancel_futures=True)
                self.executor = None
            if ready_queue is not None:
                ready_queue.close()
                ready_queue.join_thread()
            if self._temporary_directory is not None:
                self._temporary_directory.cleanup()
                self._temporary_directory = None
            raise

    def assert_compatible(self, X: np.ndarray) -> None:
        if self._input is None:
            raise RuntimeError("session input is unavailable")
        arr = np.asarray(X, dtype=np.float64)
        if arr.shape != self._input.shape:
            raise ValueError("session X differs from the matrix used to open this session")
        # The benchmark runner reuses the same immutable matrix object for each
        # fit. Avoid an O(nd) equality scan inside every fit call. For a
        # different object, verify values against the snapshot before reuse.
        if X is self._source_X:
            return
        if not np.array_equal(arr, self._input):
            raise ValueError("session X differs from the matrix used to open this session")

    def refresh_worker_info(self) -> tuple[dict[str, Any], ...]:
        """Refresh OS affinity by PID after a runner pins worker processes.

        Native thread-pool settings are captured inside each worker initializer;
        affinity is re-read from the parent process so external pinning is visible.
        """
        refreshed = []
        for worker in self.worker_info:
            info = dict(worker)
            try:
                import psutil

                info["cpu_affinity"] = psutil.Process(int(info["pid"])).cpu_affinity()
            except (ImportError, AttributeError, OSError):
                info["cpu_affinity"] = None
            refreshed.append(info)
        self.worker_info = tuple(refreshed)
        return self.worker_info

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None
        self._warmup_event = None
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
            self._temporary_directory = None
        self._input = None
        self._source_X = None
        self._entered = False


class ParallelBKM(BKM):
    """BKM using process-parallel candidate and top-r child fits.

    ``fit(X, session=...)`` reuses a persistent session. Without a session,
    ``fit(X)`` creates a one-shot session and reports setup separately from the
    core and setup-inclusive end-to-end time. Use the standard ``if __name__ ==
    '__main__'`` guard in Windows scripts.
    """

    def __init__(
        self,
        n_clusters: int,
        strategy: str = "gain",
        lambda_: float = 1.0,
        top_r: int = 3,
        n_init: int = 10,
        random_state: int = 0,
        horizon_aware: bool = False,
        n_jobs: int = 1,
    ) -> None:
        if isinstance(n_jobs, bool) or int(n_jobs) != n_jobs or int(n_jobs) < 1:
            raise ValueError("n_jobs must be a positive integer")
        super().__init__(
            n_clusters=n_clusters,
            strategy=strategy,
            lambda_=lambda_,
            top_r=top_r,
            n_init=n_init,
            random_state=random_state,
            horizon_aware=horizon_aware,
        )
        self.n_jobs = int(n_jobs)
        # Validation can opt in to retaining proposal memberships. The default
        # stays off so formal large-input fits do not retain candidate arrays.
        self._capture_proposal_trace = False
        self._capture_task_events = False
        self._behavior_context: dict[str, Any] = {}
        self._behavior_worker_pids: tuple[int, ...] = ()

    def _observe_task_batch(
        self,
        executor: ProcessPoolExecutor,
        worker: Any,
        tasks: list[tuple[Any, ...]],
        descriptors: list[dict[str, Any]],
        phase: str,
        step: int,
    ) -> tuple[list[tuple[Any, ...] | None], dict[str, Any]]:
        """Submit one parallel phase and retain per-task wall/worker observations.

        Worker durations use each worker's monotonic clock locally. Only their
        durations are compared across processes; parent submission and receipt
        events use the parent's monotonic clock and are reported separately.
        """
        phase_started = perf_counter()
        submit_started = perf_counter()
        futures: dict[Any, tuple[int, float, float]] = {}
        submit_call_seconds = 0.0
        for index, task in enumerate(tasks):
            submitted_at = perf_counter()
            future = executor.submit(worker, task)
            submit_returned_at = perf_counter()
            submit_call_seconds += submit_returned_at - submitted_at
            futures[future] = (index, submitted_at, submit_returned_at)
        submit_ended = perf_counter()

        results: list[tuple[Any, ...] | None] = [None] * len(tasks)
        errors: list[str] = []
        events: list[dict[str, Any]] = []
        fit_started = float(getattr(self, "_behavior_fit_started_at", phase_started))
        context = dict(getattr(self, "_behavior_context", {}) or {})
        for future in as_completed(futures):
            parent_received_at = perf_counter()
            index, submitted_at, submit_returned_at = futures[future]
            descriptor = descriptors[index]
            worker_meta: dict[str, Any] = {}
            result_body: tuple[Any, ...] | None = None
            error: str | None = None
            try:
                raw_result = future.result()
                worker_meta = raw_result[-1] or {}
                result_body = tuple(raw_result[:-1])
                if worker_meta.get("success") is False:
                    error = str(worker_meta.get("error") or "worker task failed")
                else:
                    # Keep the public worker return shape consistent with the
                    # executor.map path below. The observer consumes metadata
                    # for its event row, but callers still expect the trailing
                    # metadata slot when unpacking candidate/child results.
                    results[index] = tuple(raw_result)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            output_index_bytes = 0
            if phase == "primary_candidate" and result_body is not None:
                output_index_bytes = sum(
                    int(np.asarray(result_body[index]).nbytes)
                    for index in (1, 2)
                    if result_body[index] is not None
                )
            event = {
                **context,
                "level": int(step + 1),
                "phase": phase,
                "task_id": descriptor["task_id"],
                "candidate_id": descriptor.get("candidate_id"),
                "parent_candidate_id": descriptor.get("parent_candidate_id"),
                "active_candidate_count": int(getattr(self, "_current_active_candidate_count", 0)),
                "task_row_count": int(descriptor.get("task_row_count", 0)),
                "feature_count": int(getattr(self, "n_features_in_", 0) or getattr(self, "_behavior_feature_count", 0)),
                "task_input_index_bytes": int(descriptor.get("task_input_index_bytes", 0)),
                "task_output_index_bytes": output_index_bytes,
                "task_runnable": bool(descriptor.get("task_runnable", True)),
                "worker_id": worker_meta.get("worker_id"),
                "parent_submit_seconds_from_fit_start": max(0.0, submitted_at - fit_started),
                "submit_call_seconds": max(0.0, submit_returned_at - submitted_at),
                "parent_result_received_seconds_from_fit_start": max(0.0, parent_received_at - fit_started),
                "worker_compute_seconds": worker_meta.get("worker_compute_seconds"),
                "success": error is None,
                "error": error,
            }
            events.append(event)
            self.parallel_task_events_.append(event)
            if error is not None:
                errors.append(f"{descriptor['task_id']}: {error}")
        phase_ended = perf_counter()
        phase_wall = phase_ended - phase_started
        submission_wall = submit_ended - submit_started
        wait_wall = max(0.0, phase_ended - submit_ended)
        other_wall = max(0.0, phase_wall - submission_wall - wait_wall)
        durations = [
            float(event["worker_compute_seconds"])
            for event in events
            if event.get("worker_compute_seconds") is not None
        ]
        worker_load: dict[int, float] = defaultdict(float)
        for event in events:
            if event.get("worker_id") is not None and event.get("worker_compute_seconds") is not None:
                worker_load[int(event["worker_id"])] += float(event["worker_compute_seconds"])
        busy_worker_ids = sorted(worker_load)
        stats = {
            "task_count": len(tasks),
            "runnable_task_count": sum(bool(item.get("task_runnable", True)) for item in descriptors),
            "failed_task_count": len(errors),
            "phase_wall_seconds": float(phase_wall),
            "submission_wall_seconds": float(submission_wall),
            "submission_call_seconds_sum": float(submit_call_seconds),
            "wait_sync_result_seconds": float(wait_wall),
            "other_parent_seconds": float(other_wall),
            "worker_compute_seconds_sum": float(sum(durations)),
            "worker_compute_seconds_max": max(durations, default=0.0),
            "worker_compute_seconds_spread": max(durations) - min(durations) if durations else 0.0,
            "worker_count_with_tasks": len(busy_worker_ids),
            "worker_ids_with_tasks": busy_worker_ids,
            "worker_count_without_tasks": max(0, self.n_jobs - len(busy_worker_ids)),
            "longest_worker_workload_seconds": max(worker_load.values(), default=0.0),
            "max_task_over_sum_task": max(durations) / sum(durations) if durations and sum(durations) > 0 else 0.0,
            "worker_workload_seconds_by_pid": {str(pid): value for pid, value in sorted(worker_load.items())},
        }
        self._parallel_phase_stats_by_step.setdefault(step + 1, {})[phase] = stats
        if errors:
            raise RuntimeError("parallel worker task failure: " + "; ".join(errors))
        return results, stats

    def fit(
        self, X: np.ndarray, session: ParallelBKMSession | None = None
    ) -> "ParallelBKM":
        call_started = perf_counter()
        owns_session = session is None
        if owns_session:
            with ParallelBKMSession(X, n_jobs=self.n_jobs) as active_session:
                self._fit_using_pool(active_session._input, active_session.executor)
                fit_call_seconds = perf_counter() - call_started
                self.parallel_timing_ = {
                    "worker_count": self.n_jobs,
                    "session_setup_seconds": active_session.setup_seconds,
                    "core_fit_seconds": self.runtime_,
                    "fit_call_wall_seconds": fit_call_seconds,
                    "end_to_end_seconds": fit_call_seconds,
                    "session_setup_included_in_end_to_end": True,
                    "worker_pids": list(active_session.worker_pids),
                    "worker_info": list(active_session.worker_info),
                }
                active_session.fit_calls += 1
            return self

        if session.n_jobs != self.n_jobs:
            raise ValueError("estimator n_jobs must match session n_jobs")
        if not session._entered or session.executor is None:
            raise RuntimeError("session must be active in a with block")
        session.assert_compatible(X)
        self._fit_using_pool(session._input, session.executor)
        fit_call_seconds = perf_counter() - call_started
        self.parallel_timing_ = {
            "worker_count": self.n_jobs,
            "session_setup_seconds": session.setup_seconds,
            "core_fit_seconds": self.runtime_,
            "fit_call_wall_seconds": fit_call_seconds,
            "end_to_end_seconds": fit_call_seconds,
            "session_setup_included_in_end_to_end": False,
            "worker_pids": list(session.worker_pids),
            "worker_info": list(session.worker_info),
        }
        session.fit_calls += 1
        return self

    def _parallel_proposals(
        self,
        X: np.ndarray,
        candidates: Iterable[tuple[int, np.ndarray]],
        step: int,
        executor: ProcessPoolExecutor,
    ) -> dict[int, SplitProposal]:
        ordered = [(int(cid), np.asarray(members, dtype=np.int64)) for cid, members in candidates]
        tasks = [
            (cid, members, self.n_init, self._seed(step, cid), bool(self._capture_task_events))
            for cid, members in ordered
        ]
        if self._capture_task_events:
            descriptors = [
                {
                    "task_id": f"primary-l{step + 1}-c{cid}",
                    "candidate_id": cid,
                    "parent_candidate_id": None,
                    "task_row_count": int(members.size),
                    "task_input_index_bytes": int(members.nbytes),
                    "task_runnable": True,
                }
                for cid, members in ordered
            ]
            observed, _ = self._observe_task_batch(
                executor, _candidate_task, tasks, descriptors, "primary_candidate", step
            )
            results = [tuple(result) for result in observed if result is not None]
            if len(results) != len(ordered):
                raise RuntimeError("a primary candidate task returned no result")
        else:
            results = list(executor.map(_candidate_task, tasks))
        by_id: dict[int, SplitProposal] = {}
        for (cid, members), result in zip(ordered, results):
            result_id, left, right, parent_sse, child_sse, gain, _worker_meta = result
            if result_id != cid:
                raise RuntimeError("worker returned a proposal with a mismatched cluster id")
            by_id[cid] = SplitProposal(
                cluster_id=cid,
                members=members.copy(),
                left_indices=left,
                right_indices=right,
                parent_sse=parent_sse,
                child_sse=child_sse,
                gain=gain,
            )
            if getattr(self, "_capture_proposal_trace", False):
                self.proposal_trace_.append(
                    {
                        "step": int(step + 1),
                        "cluster_id": int(cid),
                        "members": members.copy(),
                        "left_indices": left.copy(),
                        "right_indices": right.copy(),
                        "parent_sse": float(parent_sse),
                        "child_sse": float(child_sse),
                        "gain": float(gain),
                    }
                )
        self.kmeans_fit_calls_ += len(by_id)
        self._step_candidate_fit_calls += len(by_id)
        self._step_candidate_fit_sample_rows += sum(int(members.size) for _, members in ordered)
        self._step_task_sizes["candidate"] = [int(members.size) for _, members in ordered]
        return by_id

    def _parallel_child_gains(
        self,
        X: np.ndarray,
        proposals: Mapping[int, SplitProposal],
        candidate_ids: Iterable[int],
        step: int,
        executor: ProcessPoolExecutor,
    ) -> dict[int, tuple[float, float]]:
        ordered_ids = list(candidate_ids)
        tasks = []
        for cid in ordered_ids:
            candidate = proposals[cid]
            for side, child in enumerate((candidate.left_indices, candidate.right_indices)):
                purpose = cid * 2 + side + 1
                tasks.append(
                    (
                        cid,
                        side,
                        child,
                        self.n_init,
                        self._seed(step + 1, purpose, 1),
                        bool(self._capture_task_events),
                    )
                )
        if self._capture_task_events:
            descriptors = [
                {
                    "task_id": f"child-l{step + 1}-c{cid}-{'left' if side == 0 else 'right'}",
                    "candidate_id": cid,
                    "parent_candidate_id": cid,
                    "task_row_count": int(members.size),
                    "task_input_index_bytes": int(members.nbytes),
                    "task_runnable": bool(members.size >= 2),
                }
                for cid, side, members, _n_init, _seed, _capture in tasks
            ]
            observed, _ = self._observe_task_batch(
                executor, _child_task, tasks, descriptors, "lookahead_child", step
            )
            results = [tuple(result) for result in observed if result is not None]
            if len(results) != len(tasks):
                raise RuntimeError("a lookahead child task returned no result")
        else:
            results = list(executor.map(_child_task, tasks))
        self._step_task_sizes["child"] = [int(task[2].size) for task in tasks]
        gains: dict[int, list[float]] = {cid: [0.0, 0.0] for cid in ordered_ids}
        fit_count = 0
        for cid, side, gain, did_fit, _worker_meta in results:
            gains[cid][side] = gain
            fit_count += did_fit
        self.kmeans_fit_calls_ += fit_count
        self._step_child_fit_calls += fit_count
        self._step_child_fit_sample_rows += sum(
            int(members.size)
            for _, _, members, _, _, _ in tasks
            if members.size >= 2
        )
        return {cid: (float(value[0]), float(value[1])) for cid, value in gains.items()}

    def _new_proposal(
        self, X: np.ndarray, cluster_id: int, members: np.ndarray, step: int
    ) -> SplitProposal:
        proposal = super()._new_proposal(X, cluster_id, members, step)
        self._step_candidate_fit_calls += 1
        self._step_candidate_fit_sample_rows += int(members.size)
        return proposal

    # The method below is a parallel-preserving copy of BKM.fit from the
    # recorded source revision. Only candidate proposal maps and top-r child
    # fit blocks use the persistent executor; selection/commit logic is kept
    # in the original order.

    def _fit_using_pool(self, X: np.ndarray, executor: ProcessPoolExecutor) -> "ParallelBKM":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must be a non-empty 2D array")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values")
        if self.n_clusters > X.shape[0]:
            raise ValueError("n_clusters cannot exceed the number of samples")

        started = perf_counter()
        self._behavior_fit_started_at = started
        self._behavior_feature_count = int(X.shape[1])
        clusters: dict[int, np.ndarray] = {0: np.arange(X.shape[0], dtype=np.int64)}
        next_cluster_id = 1
        current_sse = cluster_sse(X, np.zeros(X.shape[0], dtype=np.int64))
        self.sse_history_ = [current_sse]
        self.trajectory_: list[dict[str, Any]] = []
        self.candidate_diagnostics_: list[dict[str, Any]] = []
        self.parallel_workload_: list[dict[str, Any]] = []
        self.parallel_stage_timing_: list[dict[str, float | int]] = []
        self.parallel_task_events_: list[dict[str, Any]] = []
        self.parallel_level_tasks_: list[dict[str, Any]] = []
        self._parallel_phase_stats_by_step: dict[int, dict[str, dict[str, Any]]] = {}
        self.proposal_trace_: list[dict[str, Any]] = []
        self.n_iter_ = 0
        self.kmeans_fit_calls_ = 0

        for step in range(self.n_clusters - 1):
            step_started = perf_counter()
            self._step_task_sizes = {"candidate": [], "child": []}
            candidate_eval_seconds = 0.0
            child_lookahead_seconds = 0.0
            self._step_candidate_fit_calls = 0
            self._step_candidate_fit_sample_rows = 0
            self._step_child_fit_calls = 0
            self._step_child_fit_sample_rows = 0
            eligible = {
                cid: members for cid, members in clusters.items() if members.size >= 2
            }
            self._current_active_candidate_count = len(eligible)
            if not eligible:
                raise RuntimeError(
                    "Cannot reach the requested cluster count: every active cluster is a singleton"
                )

            proposal: SplitProposal
            future_gain: float | None = None
            score: float | None
            shortlist_size: int | None = None
            candidate_scores: dict[int, float | None] = {}
            lookahead_gains: dict[int, float | None] = {}
            lookahead_components: dict[int, tuple[float, float]] = {}
            lookahead_enabled = False

            if self.strategy in {"largest", "inertia", "mean_sse", "random"}:
                if self.strategy == "largest":
                    selected_id = min(eligible, key=lambda cid: (-eligible[cid].size, cid))
                    score = float(eligible[selected_id].size)
                    candidate_scores = {
                        cid: float(members.size) for cid, members in eligible.items()
                    }
                elif self.strategy == "inertia":
                    inertias = {
                        cid: sse_for_indices(X, members) for cid, members in eligible.items()
                    }
                    selected_id = min(inertias, key=lambda cid: (-inertias[cid], cid))
                    score = float(inertias[selected_id])
                    candidate_scores = inertias.copy()
                elif self.strategy == "mean_sse":
                    mean_sse = {
                        cid: sse_for_indices(X, members) / members.size
                        for cid, members in eligible.items()
                    }
                    selected_id = min(mean_sse, key=lambda cid: (-mean_sse[cid], cid))
                    score = float(mean_sse[selected_id])
                    candidate_scores = mean_sse.copy()
                else:
                    eligible_ids = sorted(eligible)
                    selected_id = int(
                        np.random.default_rng(self._seed(step, 0, purpose=2)).choice(
                            eligible_ids
                        )
                    )
                    score = None
                    candidate_scores = {cid: None for cid in eligible}
                proposal = self._new_proposal(
                    X, selected_id, eligible[selected_id], step
                )
            elif self.strategy == "gain":
                candidate_started = perf_counter()
                proposals = self._parallel_proposals(X, eligible.items(), step, executor)
                candidate_eval_seconds = perf_counter() - candidate_started
                selected_id = min(
                    proposals,
                    key=lambda cid: (-proposals[cid].gain, cid),
                )
                proposal = proposals[selected_id]
                score = proposal.gain
                candidate_scores = {
                    cid: candidate.gain for cid, candidate in proposals.items()
                }
            elif self.strategy == "gain_relative":
                candidate_started = perf_counter()
                proposals = self._parallel_proposals(X, eligible.items(), step, executor)
                candidate_eval_seconds = perf_counter() - candidate_started
                candidate_scores = {
                    cid: (
                        candidate.gain / candidate.parent_sse
                        if candidate.parent_sse > np.finfo(float).eps
                        else 0.0
                    )
                    for cid, candidate in proposals.items()
                }
                selected_id = min(
                    candidate_scores,
                    key=lambda cid: (-candidate_scores[cid], cid),
                )
                proposal = proposals[selected_id]
                score = candidate_scores[selected_id]
            elif self.strategy == "savaresi":
                screened_ids = sorted(
                    eligible,
                    key=lambda cid: (-eligible[cid].size, cid),
                )[:10]
                shortlist_size = len(screened_ids)
                proposals = self._parallel_proposals(
                    X, ((cid, eligible[cid]) for cid in screened_ids), step, executor
                )
                raw_scores = {
                    cid: savaresi_shape_index(X, candidate)
                    for cid, candidate in proposals.items()
                }
                finite_scores = [value for value in raw_scores.values() if value is not None]
                invalid_score = (
                    max(finite_scores) + max(1.0, abs(max(finite_scores)) * 0.05)
                    if finite_scores
                    else 1.0
                )
                candidate_scores = {
                    cid: invalid_score if value is None else value
                    for cid, value in raw_scores.items()
                }
                selected_id = min(
                    candidate_scores,
                    key=lambda cid: (candidate_scores[cid], cid),
                )
                proposal = proposals[selected_id]
                score = candidate_scores[selected_id]
            else:
                candidate_started = perf_counter()
                proposals = self._parallel_proposals(X, eligible.items(), step, executor)
                candidate_eval_seconds = perf_counter() - candidate_started
                if self.strategy in {"alss_relative", "alss_relative_screen"}:
                    screening_scores = {
                        cid: (
                            candidate.gain / candidate.parent_sse
                            if candidate.parent_sse > np.finfo(float).eps
                            else 0.0
                        )
                        for cid, candidate in proposals.items()
                    }
                else:
                    screening_scores = {
                        cid: candidate.gain for cid, candidate in proposals.items()
                    }
                screened_ids = sorted(
                    proposals,
                    key=lambda cid: (-screening_scores[cid], cid),
                )[: self.top_r]
                shortlist_size = len(screened_ids)
                scored: list[tuple[float, int, float]] = []
                lookahead_enabled = self.lambda_ > 0 and (
                    not self.horizon_aware or step < self.n_clusters - 2
                )
                if lookahead_enabled:
                    child_started = perf_counter()
                    child_gains_by_id = self._parallel_child_gains(
                        X, proposals, screened_ids, step, executor
                    )
                    child_lookahead_seconds = perf_counter() - child_started
                else:
                    child_gains_by_id = {}
                for cid in screened_ids:
                    candidate = proposals[cid]
                    if not lookahead_enabled:
                        child_gains = (0.0, 0.0)
                        candidate_future = 0.0
                    else:
                        child_gains = child_gains_by_id[cid]
                        candidate_future = (
                            sum(child_gains)
                            if self.strategy == "alss_sum"
                            else max(child_gains)
                        )
                    candidate.future_gain = candidate_future
                    if lookahead_enabled:
                        lookahead_components[cid] = tuple(map(float, child_gains))
                    candidate_score = candidate.gain + self.lambda_ * candidate_future
                    if self.strategy in {"alss_relative", "alss_relative_score"}:
                        candidate_score = (
                            candidate_score / candidate.parent_sse
                            if candidate.parent_sse > np.finfo(float).eps
                            else 0.0
                        )
                    candidate_scores[cid] = candidate_score
                    lookahead_gains[cid] = candidate_future
                    scored.append((candidate_score, cid, candidate_future))
                score, selected_id, future_gain = min(
                    scored, key=lambda item: (-item[0], item[1])
                )
                proposal = proposals[selected_id]


            if self.strategy in {"largest", "inertia", "mean_sse", "random"}:
                for cid, members in eligible.items():
                    self.candidate_diagnostics_.append(
                        {
                            "step": step + 1,
                            "strategy": self.strategy,
                            "candidate_cluster_id": cid,
                            "candidate_size": int(members.size),
                            "score_basis": self.strategy,
                            "selection_score": candidate_scores[cid],
                            "immediate_gain": proposal.gain if cid == selected_id else None,
                            "future_gain": None,
                            "candidate_evaluated": True,
                            "lookahead_evaluated": False,
                            "selected": cid == selected_id,
                            "top_r": None,
                        }
                    )
            elif self.strategy == "savaresi":
                for cid, members in eligible.items():
                    candidate = proposals.get(cid)
                    self.candidate_diagnostics_.append(
                        {
                            "step": step + 1,
                            "strategy": self.strategy,
                            "candidate_cluster_id": cid,
                            "candidate_size": int(members.size),
                            "score_basis": "savaresi_shape_index",
                            "selection_score": candidate_scores.get(cid),
                            "immediate_gain": None if candidate is None else float(candidate.gain),
                            "future_gain": None,
                            "candidate_evaluated": candidate is not None,
                            "lookahead_evaluated": False,
                            "selected": cid == selected_id,
                            "top_r": 10,
                        }
                    )
            else:
                for cid, candidate in proposals.items():
                    is_lookahead = self.strategy in {
                        "alss",
                        "alss_relative",
                        "alss_sum",
                        "alss_relative_screen",
                        "alss_relative_score",
                    }
                    in_shortlist = cid in candidate_scores if is_lookahead else False
                    score_basis = (
                        "relative_gain_plus_lookahead"
                        if self.strategy == "alss_relative"
                        else "relative_screen_absolute_lookahead"
                        if self.strategy == "alss_relative_screen"
                        else "absolute_screen_relative_lookahead"
                        if self.strategy == "alss_relative_score"
                        else "gain_plus_sum_lookahead"
                        if self.strategy == "alss_sum"
                        else "gain_plus_lookahead"
                        if self.strategy == "alss"
                        else "relative_immediate_gain"
                        if self.strategy == "gain_relative"
                        else "gain"
                    )
                    self.candidate_diagnostics_.append(
                        {
                            "step": step + 1,
                            "strategy": self.strategy,
                            "candidate_cluster_id": cid,
                            "candidate_size": int(candidate.members.size),
                            "score_basis": score_basis,
                            "selection_score": candidate_scores.get(cid),
                            "immediate_gain": float(candidate.gain),
                            "future_gain": lookahead_gains.get(cid),
                            "left_future_gain": (
                                lookahead_components[cid][0]
                                if cid in lookahead_components
                                else None
                            ),
                            "right_future_gain": (
                                lookahead_components[cid][1]
                                if cid in lookahead_components
                                else None
                            ),
                            "candidate_evaluated": True,
                            "lookahead_evaluated": in_shortlist and lookahead_enabled,
                            "lookahead_disabled_by_budget": (
                                in_shortlist
                                and self.horizon_aware
                                and self.lambda_ > 0
                                and step == self.n_clusters - 2
                            ),
                            "selected": cid == selected_id,
                            "top_r": self.top_r if is_lookahead else None,
                        }
                    )

            left_id, right_id = next_cluster_id, next_cluster_id + 1
            next_cluster_id += 2
            del clusters[selected_id]
            clusters[left_id] = proposal.left_indices
            clusters[right_id] = proposal.right_indices
            current_sse += proposal.child_sse - proposal.parent_sse

            previous = self.parallel_workload_[-1] if self.parallel_workload_ else {}
            candidate_calls = self._step_candidate_fit_calls
            child_calls = self._step_child_fit_calls
            candidate_rows = self._step_candidate_fit_sample_rows
            child_rows = self._step_child_fit_sample_rows
            self.parallel_workload_.append(
                {
                    "step": step + 1,
                    "candidate_fit_calls": candidate_calls,
                    "candidate_fit_sample_rows": candidate_rows,
                    "child_fit_calls": child_calls,
                    "child_fit_sample_rows": child_rows,
                    "total_fit_calls": candidate_calls + child_calls,
                    "total_fit_sample_rows": candidate_rows + child_rows,
                    "cumulative_candidate_fit_calls": previous.get(
                        "cumulative_candidate_fit_calls", 0
                    ) + candidate_calls,
                    "cumulative_candidate_fit_sample_rows": previous.get(
                        "cumulative_candidate_fit_sample_rows", 0
                    ) + candidate_rows,
                    "cumulative_child_fit_calls": previous.get(
                        "cumulative_child_fit_calls", 0
                    ) + child_calls,
                    "cumulative_child_fit_sample_rows": previous.get(
                        "cumulative_child_fit_sample_rows", 0
                    ) + child_rows,
                    "cumulative_total_fit_calls": previous.get(
                        "cumulative_total_fit_calls", 0
                    ) + candidate_calls + child_calls,
                    "cumulative_total_fit_sample_rows": previous.get(
                        "cumulative_total_fit_sample_rows", 0
                    ) + candidate_rows + child_rows,
                    "candidate_task_count": len(self._step_task_sizes["candidate"]),
                    "candidate_task_sizes": list(self._step_task_sizes["candidate"]),
                    "child_task_count": len(self._step_task_sizes["child"]),
                    "child_task_sizes": list(self._step_task_sizes["child"]),
                    "total_submitted_task_count": (
                        len(self._step_task_sizes["candidate"])
                        + len(self._step_task_sizes["child"])
                    ),
                    "task_input_index_bytes_estimate": int(
                        (sum(self._step_task_sizes["candidate"])
                         + sum(self._step_task_sizes["child"])) * np.dtype(np.int64).itemsize
                    ),
                    "candidate_result_index_bytes_estimate": int(
                        sum(self._step_task_sizes["candidate"])
                        * np.dtype(np.int64).itemsize
                    ),
                }
            )
            if self.parallel_workload_[-1]["cumulative_total_fit_calls"] != self.kmeans_fit_calls_:
                raise RuntimeError("parallel workload counters disagree with kmeans_fit_calls_")

            self.trajectory_.append(
                {
                    "step": step + 1,
                    "strategy": self.strategy,
                    "split_cluster_id": selected_id,
                    "cluster_size": int(proposal.members.size),
                    "left_cluster_id": left_id,
                    "right_cluster_id": right_id,
                    "left_size": int(proposal.left_indices.size),
                    "right_size": int(proposal.right_indices.size),
                    "parent_sse": float(proposal.parent_sse),
                    "children_sse": float(proposal.child_sse),
                    "immediate_gain": float(proposal.gain),
                    "future_gain": future_gain,
                    "selection_score": None if score is None else float(score),
                    "lambda": self.lambda_ if self.strategy in {
                        "alss", "alss_relative", "alss_sum",
                        "alss_relative_screen", "alss_relative_score",
                    } else None,
                    "top_r": self.top_r if self.strategy in {
                        "alss", "alss_relative", "alss_sum",
                        "alss_relative_screen", "alss_relative_score",
                    } else None,
                    "shortlist_size": shortlist_size,
                    "lookahead_horizon_aware": self.horizon_aware,
                    "lookahead_enabled": self.strategy
                    in {
                        "alss",
                        "alss_relative",
                        "alss_sum",
                        "alss_relative_screen",
                        "alss_relative_score",
                    }
                    and self.lambda_ > 0
                    and (not self.horizon_aware or step < self.n_clusters - 2),
                    "kmeans_fit_calls": self.kmeans_fit_calls_,
                    "total_sse_after_split": float(current_sse),
                }
            )
            self.sse_history_.append(float(current_sse))
            self.n_iter_ += 1
            step_total_seconds = perf_counter() - step_started
            decision_commit_seconds = max(
                0.0, step_total_seconds - candidate_eval_seconds - child_lookahead_seconds
            )
            stage_record: dict[str, Any] = {
                "step": int(step + 1),
                "step_total_seconds": float(step_total_seconds),
                "candidate_evaluation_wall_seconds": float(candidate_eval_seconds),
                "child_lookahead_wall_seconds": float(child_lookahead_seconds),
                "decision_and_commit_wall_seconds": float(decision_commit_seconds),
            }
            phase_stats = self._parallel_phase_stats_by_step.get(step + 1, {})
            primary_stats = phase_stats.get("primary_candidate", {})
            child_stats = phase_stats.get("lookahead_child", {})
            if self._capture_task_events:
                stage_record.update({
                    "primary_candidate_task_count": int(primary_stats.get("task_count", 0)),
                    "lookahead_child_task_count": int(child_stats.get("task_count", 0)),
                    "runnable_lookahead_child_task_count": int(child_stats.get("runnable_task_count", 0)),
                    "primary_candidate_submission_wall_seconds": float(primary_stats.get("submission_wall_seconds", 0.0)),
                    "primary_candidate_wait_sync_result_seconds": float(primary_stats.get("wait_sync_result_seconds", 0.0)),
                    "primary_candidate_worker_compute_seconds_sum": float(primary_stats.get("worker_compute_seconds_sum", 0.0)),
                    "primary_candidate_worker_compute_seconds_max": float(primary_stats.get("worker_compute_seconds_max", 0.0)),
                    "primary_candidate_worker_compute_seconds_spread": float(primary_stats.get("worker_compute_seconds_spread", 0.0)),
                    "primary_candidate_busy_worker_count": int(primary_stats.get("worker_count_with_tasks", 0)),
                    "primary_candidate_idle_worker_count": int(primary_stats.get("worker_count_without_tasks", self.n_jobs)),
                    "primary_candidate_longest_worker_workload_seconds": float(primary_stats.get("longest_worker_workload_seconds", 0.0)),
                    "primary_candidate_max_task_over_sum_task": float(primary_stats.get("max_task_over_sum_task", 0.0)),
                    "child_submission_wall_seconds": float(child_stats.get("submission_wall_seconds", 0.0)),
                    "child_wait_sync_result_seconds": float(child_stats.get("wait_sync_result_seconds", 0.0)),
                    "child_worker_compute_seconds_sum": float(child_stats.get("worker_compute_seconds_sum", 0.0)),
                    "child_worker_compute_seconds_max": float(child_stats.get("worker_compute_seconds_max", 0.0)),
                    "child_worker_compute_seconds_spread": float(child_stats.get("worker_compute_seconds_spread", 0.0)),
                    "child_busy_worker_count": int(child_stats.get("worker_count_with_tasks", 0)),
                    "child_idle_worker_count": int(child_stats.get("worker_count_without_tasks", self.n_jobs)),
                    "child_longest_worker_workload_seconds": float(child_stats.get("longest_worker_workload_seconds", 0.0)),
                    "child_max_task_over_sum_task": float(child_stats.get("max_task_over_sum_task", 0.0)),
                })
            self.parallel_stage_timing_.append(stage_record)
            if self._capture_task_events:
                context = dict(self._behavior_context or {})
                decision_event = {
                    **context,
                    "level": int(step + 1),
                    "phase": "decision_commit",
                    "task_id": f"decision-commit-l{step + 1}",
                    "candidate_id": int(selected_id),
                    "parent_candidate_id": int(selected_id),
                    "active_candidate_count": int(len(eligible)),
                    "task_row_count": 0,
                    "feature_count": int(X.shape[1]),
                    "task_input_index_bytes": 0,
                    "task_output_index_bytes": 0,
                    "task_runnable": True,
                    "worker_id": int(os.getpid()),
                    "worker_compute_seconds": None,
                    "parent_duration_seconds": float(decision_commit_seconds),
                    "success": True,
                    "error": None,
                    "measurement_basis": "step_total minus primary_candidate and lookahead_child phase wall times",
                }
                self.parallel_task_events_.append(decision_event)
                level_record = {
                    **context,
                    "level": int(step + 1),
                    "active_candidate_count": int(len(eligible)),
                    "primary_candidate_task_count": int(len(self._step_task_sizes["candidate"])),
                    "lookahead_child_task_count": int(len(self._step_task_sizes["child"])),
                    "runnable_lookahead_child_task_count": int(child_stats.get("runnable_task_count", 0)),
                    "candidate_phase_wall_seconds": float(candidate_eval_seconds),
                    "child_phase_wall_seconds": float(child_lookahead_seconds),
                    "decision_commit_wall_seconds": float(decision_commit_seconds),
                    "total_level_wall_seconds": float(step_total_seconds),
                    "accounting_residual_seconds": float(
                        step_total_seconds - candidate_eval_seconds - child_lookahead_seconds - decision_commit_seconds
                    ),
                    "candidate_submission_wall_seconds": float(primary_stats.get("submission_wall_seconds", 0.0)),
                    "candidate_wait_sync_result_seconds": float(primary_stats.get("wait_sync_result_seconds", 0.0)),
                    "candidate_worker_compute_seconds_sum": float(primary_stats.get("worker_compute_seconds_sum", 0.0)),
                    "candidate_worker_compute_seconds_max": float(primary_stats.get("worker_compute_seconds_max", 0.0)),
                    "candidate_task_duration_spread_seconds": float(primary_stats.get("worker_compute_seconds_spread", 0.0)),
                    "candidate_busy_worker_count": int(primary_stats.get("worker_count_with_tasks", 0)),
                    "candidate_idle_worker_count": int(primary_stats.get("worker_count_without_tasks", self.n_jobs)),
                    "candidate_longest_worker_workload_seconds": float(primary_stats.get("longest_worker_workload_seconds", 0.0)),
                    "candidate_max_task_over_sum_task": float(primary_stats.get("max_task_over_sum_task", 0.0)),
                    "candidate_worker_ids_json": json.dumps(primary_stats.get("worker_ids_with_tasks", [])),
                    "child_submission_wall_seconds": float(child_stats.get("submission_wall_seconds", 0.0)),
                    "child_wait_sync_result_seconds": float(child_stats.get("wait_sync_result_seconds", 0.0)),
                    "child_worker_compute_seconds_sum": float(child_stats.get("worker_compute_seconds_sum", 0.0)),
                    "child_worker_compute_seconds_max": float(child_stats.get("worker_compute_seconds_max", 0.0)),
                    "child_task_duration_spread_seconds": float(child_stats.get("worker_compute_seconds_spread", 0.0)),
                    "child_busy_worker_count": int(child_stats.get("worker_count_with_tasks", 0)),
                    "child_idle_worker_count": int(child_stats.get("worker_count_without_tasks", self.n_jobs)),
                    "child_longest_worker_workload_seconds": float(child_stats.get("longest_worker_workload_seconds", 0.0)),
                    "child_max_task_over_sum_task": float(child_stats.get("max_task_over_sum_task", 0.0)),
                    "child_worker_ids_json": json.dumps(child_stats.get("worker_ids_with_tasks", [])),
                    "task_input_index_bytes_estimate": int(
                        (sum(self._step_task_sizes["candidate"]) + sum(self._step_task_sizes["child"]))
                        * np.dtype(np.int64).itemsize
                    ),
                    "candidate_result_index_bytes_estimate": int(
                        sum(self._step_task_sizes["candidate"]) * np.dtype(np.int64).itemsize
                    ),
                    "candidate_fit_calls": int(self._step_candidate_fit_calls),
                    "child_fit_calls": int(self._step_child_fit_calls),
                    "total_fit_calls": int(self._step_candidate_fit_calls + self._step_child_fit_calls),
                    "selected_parent_id": int(selected_id),
                    "decision_commit_measurement": "remainder after measured parallel phase wall times; phases can include overlapping worker compute and parent wait",
                }
                self.parallel_level_tasks_.append(level_record)

        self.cluster_indices_ = {
            cid: members.copy() for cid, members in sorted(clusters.items())
        }
        last_work = self.parallel_workload_[-1] if self.parallel_workload_ else {}
        self.fit_work_by_step_ = list(self.parallel_workload_)
        self.fit_work_ = {
            "candidate_fit_calls": last_work.get("cumulative_candidate_fit_calls", 0),
            "candidate_fit_sample_rows": last_work.get(
                "cumulative_candidate_fit_sample_rows", 0
            ),
            "child_fit_calls": last_work.get("cumulative_child_fit_calls", 0),
            "child_fit_sample_rows": last_work.get(
                "cumulative_child_fit_sample_rows", 0
            ),
            "total_fit_calls": last_work.get("cumulative_total_fit_calls", 0),
            "total_fit_sample_rows": last_work.get(
                "cumulative_total_fit_sample_rows", 0
            ),
        }
        self.labels_ = np.empty(X.shape[0], dtype=np.int64)
        for label, members in enumerate(self.cluster_indices_.values()):
            self.labels_[members] = label
        self.sse_ = cluster_sse(X, self.labels_)
        self.runtime_ = perf_counter() - started
        self.n_features_in_ = X.shape[1]
        return self

