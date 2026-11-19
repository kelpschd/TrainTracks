"""Compute resources for a run: how many movies in parallel, threads per movie, memory per movie.

The pipeline's unit of work is one movie. How many run at once depends on the machine, so it is
described, not hard-coded. Each field is resolved in this order (first one that is set wins):

1. explicit argument (CLI flag),
2. the scheduler environment (SLURM: ``SLURM_CPUS_PER_TASK`` for threads; inside a job array each
   array task runs one movie at a time),
3. a named profile in ``TrainTracks.toml`` (``--profile NAME`` -> ``[profiles.NAME]``),
4. the ``[resources]`` table of ``TrainTracks.toml``,
5. automatic: the largest ``jobs <= performance cores // 2`` whose total estimated memory fits in
   75% of RAM, with ``threads_per_job = performance cores // jobs``.

Memory per job depends on its threads, because detection holds one frame's blob-search arrays per
thread. Measured on 33-frame 2048x2048x4 movies (2026-11-19): peak 3.3-3.5 GB at 2 threads, 8.8-9.5 GB
at 8 threads, i.e. about ``1.6 + 0.9 * threads`` GB. That model is used unless ``mem_per_job_gb`` is
set (flag, profile or ``[resources]``), which pins a fixed estimate instead.

Example ``TrainTracks.toml``::

    [resources]
    mem_per_job_gb = 4        # optional: pin a fixed estimate instead of the threads-based model

    [profiles.laptop]
    jobs = 4
    threads_per_job = 2

    [profiles.hpc_array]      # one movie per SLURM array task
    jobs = 1
    threads_per_job = 8

Each run records the real peak memory (``peak_rss_gb``) and stage timings of every movie in its
metadata, so the model can be checked against new data or new hardware.

Light imports only (no numpy), so worker processes can import this before any numeric library.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass

MIN_THREADS_PER_JOB = 2
RAM_FRACTION = 0.75
MEM_BASE_GB = 1.6          # peak memory model, measured 2026-11-19 (see module docstring)
MEM_PER_THREAD_GB = 0.9


@dataclass(frozen=True)
class Hardware:
    logical_cores: int
    performance_cores: int
    memory_gb: float


@dataclass(frozen=True)
class Resources:
    jobs: int
    threads_per_job: int
    mem_per_job_gb: float      # estimate used for sizing (modelled from threads unless pinned)


def _macos_performance_cores() -> int | None:
    if sys.platform != "darwin":
        return None
    try:
        out = subprocess.run(
            ["sysctl", "-n", "hw.perflevel0.physicalcpu"], capture_output=True, text=True, timeout=5
        )
        return int(out.stdout.strip()) if out.returncode == 0 and out.stdout.strip() else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _logical_cores() -> int:
    if hasattr(os, "sched_getaffinity"):  # respects cpusets / SLURM allocations on Linux
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)


def _memory_gb(env: dict) -> float:
    # A scheduler allocation is the real limit on a cluster node.
    if env.get("SLURM_MEM_PER_NODE"):
        return float(env["SLURM_MEM_PER_NODE"]) / 1024.0  # MB -> GB
    if env.get("SLURM_MEM_PER_CPU") and env.get("SLURM_CPUS_PER_TASK"):
        return float(env["SLURM_MEM_PER_CPU"]) * float(env["SLURM_CPUS_PER_TASK"]) / 1024.0
    try:
        import psutil

        return psutil.virtual_memory().total / 1024**3
    except ImportError:
        pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    except (ValueError, OSError, AttributeError):
        return 8.0


def detect_hardware(env: dict | None = None) -> Hardware:
    """Cores (performance cores on Apple Silicon, so efficiency cores are not counted) and RAM."""
    env = os.environ if env is None else env
    logical = _logical_cores()
    perf = _macos_performance_cores()
    if env.get("SLURM_CPUS_PER_TASK"):
        perf = int(env["SLURM_CPUS_PER_TASK"])
    perf = min(perf, logical) if perf else logical
    return Hardware(logical_cores=logical, performance_cores=perf, memory_gb=_memory_gb(env))


def _parse_jobs(value) -> int | None:
    if value is None or str(value).lower() == "auto":
        return None
    n = int(value)
    if n < 1:
        raise ValueError(f"jobs must be >= 1 or 'auto', got {value!r}")
    return n


def estimated_mem_gb(threads: int) -> float:
    """Modelled peak memory of one movie running with ``threads`` threads."""
    return MEM_BASE_GB + MEM_PER_THREAD_GB * threads


def auto_jobs(hw: Hardware, mem_per_job_gb: float | None = None, min_threads: int = MIN_THREADS_PER_JOB) -> int:
    """Largest number of parallel movies that fits the cores and 75% of RAM.

    ``mem_per_job_gb=None`` uses the threads-based model; a number pins a fixed estimate.
    """
    by_cpu = max(1, hw.performance_cores // min_threads)
    for jobs in range(by_cpu, 0, -1):
        threads = max(1, hw.performance_cores // jobs)
        mem = mem_per_job_gb if mem_per_job_gb is not None else estimated_mem_gb(threads)
        if jobs * mem <= RAM_FRACTION * hw.memory_gb:
            return jobs
    return 1


def resolve_resources(
    profile: str | None = None,
    jobs=None,
    threads_per_job: int | None = None,
    mem_per_job_gb: float | None = None,
    config_file=None,
    hardware: Hardware | None = None,
    env: dict | None = None,
    toml_data: dict | None = None,
) -> Resources:
    """Resolve ``Resources`` (see module docstring for the priority order)."""
    env = os.environ if env is None else env
    hw = hardware or detect_hardware(env)
    if toml_data is None:
        from TrainTracks.config import find_config_file, load_toml

        toml_data = load_toml(find_config_file(config_file))
    table = dict(toml_data.get("resources", {}))
    if profile:
        profiles = toml_data.get("profiles", {})
        if profile not in profiles:
            raise KeyError(f"profile {profile!r} not found in config (have: {sorted(profiles)})")
        table.update(profiles[profile])

    pinned = mem_per_job_gb if mem_per_job_gb is not None else table.get("mem_per_job_gb")
    pinned = float(pinned) if pinned is not None else None

    n_jobs = _parse_jobs(jobs)
    if n_jobs is None and env.get("SLURM_ARRAY_TASK_ID") is not None and "jobs" not in table:
        n_jobs = 1  # a SLURM array task is one worker; parallelism comes from the array
    if n_jobs is None:
        n_jobs = _parse_jobs(table.get("jobs"))
    if n_jobs is None:
        n_jobs = auto_jobs(hw, pinned)

    threads = threads_per_job
    if threads is None and env.get("SLURM_CPUS_PER_TASK"):
        threads = max(1, int(env["SLURM_CPUS_PER_TASK"]) // n_jobs)
    if threads is None and "threads_per_job" in table:
        threads = int(table["threads_per_job"])
    if threads is None:
        threads = max(1, hw.performance_cores // n_jobs)
    mem = pinned if pinned is not None else estimated_mem_gb(int(threads))
    return Resources(jobs=n_jobs, threads_per_job=int(threads), mem_per_job_gb=round(mem, 2))


def apply_thread_limits(threads: int) -> None:
    """Cap the threads used by BLAS/OpenMP, OpenCV and dask in this process.

    Parallelism only changes speed, not results. Environment variables only take effect for
    libraries imported afterwards, so worker processes call this first thing.
    """
    n = str(max(1, int(threads)))
    for var in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS",
    ):
        os.environ[var] = n
    try:
        import cv2

        cv2.setNumThreads(int(n))
    except ImportError:
        pass
    try:
        import dask

        dask.config.set(num_workers=int(n))
    except ImportError:
        pass
