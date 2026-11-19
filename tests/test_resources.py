"""Tests for TrainTracks.resources (resolution order, auto sizing, SLURM environment)."""
from TrainTracks.resources import Hardware, auto_jobs, resolve_resources

HW = Hardware(logical_cores=12, performance_cores=8, memory_gb=32.0)  # the 12-core / 32 GB Mac


def _res(**kw):
    kw.setdefault("hardware", HW)
    kw.setdefault("env", {})
    kw.setdefault("toml_data", {})
    return resolve_resources(**kw)


def test_auto_is_limited_by_memory_then_cores():
    assert auto_jobs(HW, mem_per_job_gb=8.0) == 3  # 0.75 * 32 // 8
    assert auto_jobs(HW, mem_per_job_gb=4.0) == 4  # memory allows 6, performance cores // 2 = 4
    assert auto_jobs(Hardware(4, 2, 4.0), mem_per_job_gb=8.0) == 1  # never below 1


def test_auto_threads_use_the_performance_cores():
    r = _res()
    assert (r.jobs, r.threads_per_job) == (3, 2)
    assert _res(jobs=1).threads_per_job == 8


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
