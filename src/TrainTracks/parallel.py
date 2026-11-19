"""Run independent jobs (one per movie) serially, in a local process pool, or as a cluster shard.

Design goals:

* The same job function runs everywhere: serial (``jobs=1``, in this process, easy to debug),
  a local process pool (``jobs>1``), or one SLURM array task each (``shard`` / ``shard_from_env``).
* One failing movie never stops the others (unless ``stop_on_error``).
* With a pool, each movie runs in a fresh process (so peak memory per movie is exact and memory is
  returned to the OS), and its output goes to ``<log_dir>/<name>.log`` instead of interleaving.

Light imports only; worker processes import this before any numeric library.
"""

from __future__ import annotations

import contextlib
import multiprocessing
import os
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from TrainTracks.resources import apply_thread_limits


@dataclass
class JobResult:
    item: Any
    ok: bool
    value: Any = None
    error: str | None = None
    traceback: str | None = None
    seconds: float = 0.0


def shard(items: Sequence, index: int, count: int) -> list:
    """Round-robin slice ``items[index::count]``. ``items`` must be in a deterministic order."""
    if count < 1 or not 0 <= index < count:
        raise ValueError(f"bad shard {index}/{count}")
    return list(items)[index::count]


def shard_from_env(env: dict | None = None) -> tuple[int, int]:
    """(index, count) from a SLURM job array if present, else (0, 1).

    Uses ``SLURM_ARRAY_TASK_ID`` and ``SLURM_ARRAY_TASK_COUNT``. Array ids are assumed to be
    0..count-1 (``#SBATCH --array=0-N``); for other id schemes pass the shard explicitly.
    """
    env = os.environ if env is None else env
    if env.get("SLURM_ARRAY_TASK_ID") is not None and env.get("SLURM_ARRAY_TASK_COUNT") is not None:
        return int(env["SLURM_ARRAY_TASK_ID"]), int(env["SLURM_ARRAY_TASK_COUNT"])
    return 0, 1


def _init_worker(threads: int | None) -> None:
    if threads:
        apply_thread_limits(threads)


def _run_one(func: Callable, item: Any, log_path: str | None) -> JobResult:
    t0 = time.perf_counter()
    try:
        if log_path:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w", buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                value = func(item)
        else:
            value = func(item)
        return JobResult(item, True, value=value, seconds=time.perf_counter() - t0)
    except Exception as exc:  # noqa: BLE001 - reported to the caller, not swallowed
        tb = traceback.format_exc()
        if log_path:
            with open(log_path, "a") as log:
                log.write("\n" + tb)
        return JobResult(item, False, error=f"{type(exc).__name__}: {exc}", traceback=tb,
                         seconds=time.perf_counter() - t0)


def run_jobs(
    func: Callable[[Any], Any],
    items: Sequence,
    jobs: int = 1,
    threads_per_job: int | None = None,
    log_dir: Path | str | None = None,
    name_of: Callable[[Any], str] = str,
    stop_on_error: bool = False,
    on_done: Callable[[JobResult, int, int], None] | None = None,
) -> list[JobResult]:
    """Run ``func(item)`` for every item; return one ``JobResult`` per item in input order.

    ``jobs=1`` runs in this process and prints normally. ``jobs>1`` uses a spawn-based process
    pool (``func`` must be importable at module level) with one fresh process per item and a log
    file per item in ``log_dir``. ``on_done(result, n_done, n_total)`` is called as items finish.
    """
    items = list(items)
    results: dict[int, JobResult] = {}
    total = len(items)

    if jobs <= 1:
        if threads_per_job:
            apply_thread_limits(threads_per_job)
        for i, item in enumerate(items):
            r = _run_one(func, item, None)
            results[i] = r
            if on_done:
                on_done(r, len(results), total)
            if not r.ok and stop_on_error:
                break
        return [results[i] for i in sorted(results)]

    kwargs: dict = {}
    if sys.version_info >= (3, 11):
        kwargs["max_tasks_per_child"] = 1  # fresh process per movie: exact peak memory, memory freed
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx, initializer=_init_worker,
                             initargs=(threads_per_job,), **kwargs) as ex:
        pending = {}
        todo = list(enumerate(items))
        stopped = False

        def submit_next() -> None:
            i, item = todo.pop(0)
            log = str(Path(log_dir) / f"{name_of(item)}.log") if log_dir else None
            pending[ex.submit(_run_one, func, item, log)] = i

        while todo and len(pending) < jobs:
            submit_next()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                i = pending.pop(fut)
                r = fut.result()
                results[i] = r
                if on_done:
                    on_done(r, len(results), total)
                if not r.ok and stop_on_error:
                    stopped = True
            while todo and len(pending) < jobs and not stopped:
                submit_next()
    return [results[i] for i in sorted(results)]
