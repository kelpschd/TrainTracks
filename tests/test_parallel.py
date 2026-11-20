"""Tests for TrainTracks.parallel. Worker functions are module-level so spawn can import them."""
from TrainTracks.parallel import run_jobs, shard, shard_from_env


def square(x):
    return x * x


def fail_on_three(x):
    if x == 3:
        raise ValueError("three is unlucky")
    print(f"working on {x}")
    return x


def test_shard_is_a_deterministic_partition():
    items = list(range(10))
    parts = [shard(items, i, 3) for i in range(3)]
    assert sorted(sum(parts, [])) == items and parts[0] == [0, 3, 6, 9]


def test_shard_from_env():
    assert shard_from_env({}) == (0, 1)
    assert shard_from_env({"SLURM_ARRAY_TASK_ID": "2", "SLURM_ARRAY_TASK_COUNT": "8"}) == (2, 8)


def test_serial_runs_in_order_and_isolates_failures():
    res = run_jobs(fail_on_three, [1, 2, 3, 4], jobs=1)
    assert [r.ok for r in res] == [True, True, False, True]
    assert res[2].error.startswith("ValueError") and "three is unlucky" in res[2].error
    assert [r.value for r in res if r.ok] == [1, 2, 4]


def test_stop_on_error_serial():
    res = run_jobs(fail_on_three, [1, 3, 5], jobs=1, stop_on_error=True)
    assert len(res) == 2 and not res[1].ok


def test_pool_matches_serial_and_writes_logs(tmp_path):
    done = []
    res = run_jobs(fail_on_three, [1, 2, 3, 4, 5], jobs=2, log_dir=tmp_path, name_of=lambda x: f"m{x}",
                   on_done=lambda r, n, total: done.append((n, total)))
    assert [r.item for r in res] == [1, 2, 3, 4, 5]  # input order, whatever order they finished
    assert [r.ok for r in res] == [True, True, False, True, True]
    assert sorted(done) == [(i, 5) for i in range(1, 6)]
    assert "working on 2" in (tmp_path / "m2.log").read_text()
    assert "three is unlucky" in (tmp_path / "m3.log").read_text()


def test_pool_returns_values():
    assert [r.value for r in run_jobs(square, [2, 3, 4], jobs=3)] == [4, 9, 16]
