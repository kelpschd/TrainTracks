"""Tests for TrainTracks.resources (resolution order, auto sizing, SLURM environment)."""
from TrainTracks.resources import Hardware, auto_jobs, estimated_mem_gb, peak_rss_gb, resolve_resources

HW = Hardware(logical_cores=12, performance_cores=8, memory_gb=32.0)  # the 12-core / 32 GB Mac


def _res(**kw):
    kw.setdefault("hardware", HW)
    kw.setdefault("env", {})
    kw.setdefault("toml_data", {})
    return resolve_resources(**kw)


def test_pinned_memory_limits_jobs_then_cores():
    assert auto_jobs(HW, mem_per_job_gb=8.0) == 3  # 0.75 * 32 // 8
    assert auto_jobs(HW, mem_per_job_gb=4.0) == 4  # memory allows 6, performance cores // 2 = 4
    assert auto_jobs(Hardware(4, 2, 4.0), mem_per_job_gb=8.0) == 1  # never below 1


def test_threads_based_memory_model_covers_the_measurements():
    # measured 2026-11-19: peaks 3.0-4.6 GB at 2 threads (32 movies), 8.8-9.5 GB at 8 threads
    assert 4.5 <= estimated_mem_gb(2) <= 5.0 and 9.5 <= estimated_mem_gb(8) <= 10.5
    assert auto_jobs(HW) == 4                                    # 12-core / 32 GB Mac: 4 jobs x 2 threads
    assert auto_jobs(Hardware(8, 8, 24.0)) == 3                  # 24 GB: 4 jobs would need 18.4 > 0.75 * 24
    assert auto_jobs(Hardware(64, 64, 256.0)) == 32              # a big node: limited by cores // 2


def test_auto_threads_use_the_performance_cores():
    r = _res()
    assert (r.jobs, r.threads_per_job) == (4, 2) and 4.5 <= r.mem_per_job_gb <= 5.0
    assert _res(jobs=1).threads_per_job == 8
    assert _res(mem_per_job_gb=8).jobs == 3


def test_cli_beats_profile_beats_resources_table():
    toml = {"resources": {"jobs": 2, "threads_per_job": 4, "mem_per_job_gb": 6},
            "profiles": {"laptop": {"jobs": 3, "threads_per_job": 2}}}
    assert _res(toml_data=toml).jobs == 2
    assert _res(toml_data=toml, profile="laptop").jobs == 3
    r = _res(toml_data=toml, profile="laptop", jobs="5", threads_per_job=1)
    assert (r.jobs, r.threads_per_job) == (5, 1)
    assert _res(toml_data=toml).mem_per_job_gb == 6.0


def test_unknown_profile_is_an_error():
    try:
        _res(profile="nope", toml_data={"profiles": {"laptop": {}}})
    except KeyError as e:
        assert "nope" in str(e)
    else:
        raise AssertionError("expected KeyError")


def test_slurm_array_task_runs_one_movie_with_all_allocated_cpus():
    env = {"SLURM_ARRAY_TASK_ID": "3", "SLURM_CPUS_PER_TASK": "8"}
    r = _res(env=env)
    assert (r.jobs, r.threads_per_job) == (1, 8)


def test_bad_jobs_value_is_rejected():
    try:
        _res(jobs=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_single_threaded_jobs_are_limited_by_memory_not_cores():
    # the segmentation script: 1 thread per movie, 6 GB pinned -> 4 jobs fit in 0.75 * 32 GB, though 8 cores are free
    r = _res(jobs="auto", threads_per_job=1, mem_per_job_gb=6.0, min_threads_per_job=1)
    assert (r.jobs, r.threads_per_job) == (4, 1)
    assert _res(jobs="auto", threads_per_job=1, mem_per_job_gb=2.0, min_threads_per_job=1).jobs == 8  # all performance cores


def test_peak_rss_is_a_positive_number_of_gb():
    v = peak_rss_gb()
    assert v is None or 0.0 < v < 1024.0
