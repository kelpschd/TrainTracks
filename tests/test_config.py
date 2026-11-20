"""Tests for TrainTracks.config (paths resolution and stored-path handling)."""
from pathlib import Path

import pytest

from TrainTracks import config
from TrainTracks.config import (
    default_raw_root,
    load_paths,
    resolve_input_zarr,
    resolve_stored_path,
    to_relative,
)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    for var in (config.ENV_RAW, config.ENV_RESULTS, config.ENV_CONFIG):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "_repo_root", lambda: tmp_path / "no_repo_here")


def test_missing_paths_raise_or_return_none():
    with pytest.raises(ValueError, match="raw_root and results_root not set"):
        load_paths()
    assert load_paths(require=False) is None


def test_explicit_args_beat_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(config.ENV_RAW, str(tmp_path / "env_raw"))
    monkeypatch.setenv(config.ENV_RESULTS, str(tmp_path / "env_res"))
    p = load_paths(raw_root=tmp_path / "arg_raw")
    assert p.raw_root == (tmp_path / "arg_raw").resolve()
    assert p.results_root == (tmp_path / "env_res").resolve()


def test_toml_file_is_read_and_env_overrides_it(monkeypatch, tmp_path):
    pytest.importorskip("tomllib")
    (tmp_path / "TrainTracks.toml").write_text(
        f'[paths]\nraw_root = "{tmp_path}/f_raw"\nresults_root = "{tmp_path}/f_res"\n'
    )
    p = load_paths()
    assert p.raw_root == (tmp_path / "f_raw").resolve() and p.results_root == (tmp_path / "f_res").resolve()
    monkeypatch.setenv(config.ENV_RESULTS, str(tmp_path / "env_res"))
    assert load_paths().results_root == (tmp_path / "env_res").resolve()


def test_explicit_config_file_must_exist(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_paths(config_file=tmp_path / "nope.toml")


def test_relative_path_helpers(tmp_path):
    root = tmp_path / "results"
    assert to_relative(root / "m1" / "blobs.csv", root) == "m1/blobs.csv"
    assert to_relative(tmp_path / "elsewhere" / "x", root) == str(tmp_path / "elsewhere" / "x")
    assert resolve_stored_path("m1/blobs.csv", root) == root / "m1" / "blobs.csv"
    assert resolve_stored_path("/abs/old/path.csv", root) == Path("/abs/old/path.csv")


def _dataset(tmp_path, movie="m1"):
    (tmp_path / "ds" / "raw_data" / f"{movie}.zarr").mkdir(parents=True)
    movie_dir = tmp_path / "ds" / "results" / movie
    movie_dir.mkdir(parents=True)
    return movie_dir


def test_default_raw_root_is_sibling_of_results(tmp_path):
    movie_dir = _dataset(tmp_path)
    assert default_raw_root(movie_dir) == (tmp_path / "ds" / "raw_data").resolve()


def test_input_zarr_stale_absolute_path_falls_back_to_sibling_raw_data(tmp_path):
    movie_dir = _dataset(tmp_path)
    meta = {"input_zarr": "/old/place/raw_data/m1.zarr"}  # stale, does not exist
    assert resolve_input_zarr(meta, movie_dir) == (tmp_path / "ds" / "raw_data").resolve() / "m1.zarr"


def test_input_zarr_relative_name_and_explicit_raw_root(tmp_path):
    movie_dir = _dataset(tmp_path)
    other = tmp_path / "other_raw"
    (other / "m1.zarr").mkdir(parents=True)
    assert resolve_input_zarr({"input_zarr": "m1.zarr"}, movie_dir, raw_root=other) == other / "m1.zarr"
    assert resolve_input_zarr({}, movie_dir) == (tmp_path / "ds" / "raw_data").resolve() / "m1.zarr"


def test_input_zarr_existing_absolute_path_wins(tmp_path):
    movie_dir = _dataset(tmp_path)
    existing = tmp_path / "ds" / "raw_data" / "m1.zarr"
    assert resolve_input_zarr({"input_zarr": str(existing)}, movie_dir) == existing


def test_input_zarr_not_found_lists_places_tried(tmp_path):
    movie_dir = tmp_path / "ds" / "results" / "ghost"
    movie_dir.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="ghost.zarr"):
        resolve_input_zarr({}, movie_dir)
