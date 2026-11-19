"""Where the data lives: ``raw_root`` and ``results_root``.

Dataset layout (decided 2026-11-19), both folders side by side under the dataset root::

    <dataset>/raw_data/   the .nd2 files and their .zarr conversions (nothing else)
    <dataset>/results/    EVERY pipeline output, zarr and csv, one folder per movie

Nothing in the repo should hard-code those paths. ``load_paths`` resolves them from, in
order of priority:

1. explicit arguments (CLI flags),
2. the environment: ``TRAINTRACKS_RAW_ROOT`` / ``TRAINTRACKS_RESULTS_ROOT``,
3. a TOML file: ``$TRAINTRACKS_CONFIG``, else ``TrainTracks.toml`` in the current
   directory or the repo root (copy ``TrainTracks.example.toml`` and edit it).

``metadata.json`` stores paths RELATIVE to these roots (``input_zarr`` relative to
``raw_root``, ``output_*`` relative to ``results_root``) so moving or renaming a folder
cannot leave stale absolute paths behind. ``resolve_stored_path`` and
``resolve_input_zarr`` read both the new relative form and the old absolute form.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ENV_RAW = "TRAINTRACKS_RAW_ROOT"
ENV_RESULTS = "TRAINTRACKS_RESULTS_ROOT"
ENV_CONFIG = "TRAINTRACKS_CONFIG"
CONFIG_NAME = "TrainTracks.toml"
RAW_DIRNAME = "raw_data"


@dataclass(frozen=True)
class Paths:
    raw_root: Path
    results_root: Path


def _repo_root() -> Path:
    # src/TrainTracks/config.py -> repo root is two levels above the package folder
    return Path(__file__).resolve().parents[2]


def find_config_file(explicit: Path | str | None = None) -> Path | None:
    """Return the config file to use, or None. An explicit/env path must exist."""
    if explicit is None and os.environ.get(ENV_CONFIG):
        explicit = os.environ[ENV_CONFIG]
    if explicit is not None:
        p = Path(explicit).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"Config file not found: {p}")
        return p
    for folder in (Path.cwd(), _repo_root()):
        p = folder / CONFIG_NAME
        if p.is_file():
            return p
    return None


def _from_file(path: Path | None) -> dict:
    if path is None:
        return {}
    import tomllib  # stdlib in Python >= 3.11 (the project's required version)

    with path.open("rb") as f:
        data = tomllib.load(f)
    return data.get("paths", data)


def load_paths(
    raw_root: Path | str | None = None,
    results_root: Path | str | None = None,
    config_file: Path | str | None = None,
    require: bool = True,
) -> Paths | None:
    """Resolve ``raw_root`` and ``results_root`` (see module docstring for priority).

    With ``require=False`` returns None instead of raising when either is unset.
    """
    file_vals = _from_file(find_config_file(config_file))
    raw = raw_root or os.environ.get(ENV_RAW) or file_vals.get("raw_root")
    res = results_root or os.environ.get(ENV_RESULTS) or file_vals.get("results_root")
    if not raw or not res:
        if not require:
            return None
        missing = [n for n, v in (("raw_root", raw), ("results_root", res)) if not v]
        raise ValueError(
            f"{' and '.join(missing)} not set. Pass the CLI flag, set "
            f"{ENV_RAW}/{ENV_RESULTS}, or create {CONFIG_NAME} (see TrainTracks.example.toml)."
        )
    return Paths(Path(raw).expanduser().resolve(), Path(res).expanduser().resolve())


def movie_results_dir(paths: Paths, movie: str) -> Path:
    return paths.results_root / movie


def raw_zarr_path(paths: Paths, movie: str) -> Path:
    return paths.raw_root / f"{movie}.zarr"


def to_relative(path: Path | str, root: Path | str) -> str:
    """POSIX path of ``path`` relative to ``root``; the absolute path if it is not under it."""
    p, r = Path(path), Path(root)
    try:
        return p.resolve().relative_to(r.resolve()).as_posix()
    except ValueError:
        return str(p)


def resolve_stored_path(value: str | Path, root: Path | str) -> Path:
    """A path read from metadata.json: absolute (old files) stays, relative joins ``root``."""
    p = Path(value)
    return p if p.is_absolute() else Path(root) / p


def default_raw_root(movie_dir: Path | str) -> Path:
    """Sibling-folder convention: <dataset>/results/<movie> -> <dataset>/raw_data."""
    return Path(movie_dir).resolve().parent.parent / RAW_DIRNAME


def resolve_input_zarr(meta: dict, movie_dir: Path | str, raw_root: Path | str | None = None) -> Path:
    """Find the raw ``<movie>.zarr`` for a results movie folder.

    Uses ``meta['input_zarr']`` if it is an existing absolute path (old metadata);
    otherwise looks up its file name (or ``<movie>.zarr``) under, in order: ``raw_root``
    argument, the configured ``raw_root``, and ``<dataset>/raw_data`` next to the results
    folder. Stale absolute paths in old metadata therefore do not matter.
    """
    movie_dir = Path(movie_dir)
    name = f"{movie_dir.name}.zarr"
    stored = meta.get("input_zarr")
    if stored:
        p = Path(stored)
        if p.is_absolute() and p.exists():
            return p
        name = p.name
    roots: list[Path] = []
    if raw_root:
        roots.append(Path(raw_root).expanduser())
    cfg = load_paths(require=False)
    if cfg is not None:
        roots.append(cfg.raw_root)
    roots.append(default_raw_root(movie_dir))
    tried = []
    for root in roots:
        cand = root / name
        tried.append(str(cand))
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Raw zarr for {movie_dir.name} not found; tried: {tried}")
