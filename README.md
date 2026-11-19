# TrainTracks

## Getting Started

Create a new environment
```bash
conda create -n TrainTracks python=3.11
conda activate TrainTracks
pip install -e .
```

## Configuration

Data lives outside the repo, in two folders side by side under the dataset root:

- `raw_data/`: the `.nd2` files and their `.zarr` conversions (nothing else).
- `results/`: every pipeline output (zarr and csv), one folder per movie.

Tell the code where they are with `TrainTracks.toml` (copy `TrainTracks.example.toml`; it is
gitignored), the environment variables `TRAINTRACKS_RAW_ROOT` / `TRAINTRACKS_RESULTS_ROOT`,
or CLI flags (highest priority). `metadata.json` stores paths relative to these roots, so
moving a folder never leaves stale absolute paths.

Checks: `pytest tests`, and `python scripts/check_regression.py detection|flow|tracking
<movie_dir>` re-runs a pipeline stage and compares it with the stored outputs.

Parallel and cluster runs: see `docs/parallel_and_hpc.md` (`batch_pipeline.py --jobs`, `--profile`, SLURM arrays).
