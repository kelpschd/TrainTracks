# Running the pipeline in parallel (laptop now, cluster later)

The unit of work is **one movie**. Movies are independent, so the same code runs serially, in a local
process pool, or as one cluster task per movie. Parallelism changes speed only, never results
(check with `scripts/compare_runs.py`).

## On one machine

```bash
python scripts/batch_pipeline.py                       # sizes itself from the machine
python scripts/batch_pipeline.py --jobs 3              # 3 movies at once
python scripts/batch_pipeline.py --profile laptop      # [profiles.laptop] in TrainTracks.toml
python scripts/batch_pipeline.py --movie sg100_Well5_1018 --output-dir /tmp/scratch
python scripts/batch_pipeline.py --skip-existing       # resume: skip movies that are already complete
```

Automatic sizing: `jobs = min(performance cores // 2, 0.75 * RAM // mem_per_job_gb)` and
`threads_per_job = performance cores // jobs` (Apple Silicon efficiency cores are not counted).
`mem_per_job_gb` is only a cap (default 8). Every movie's `metadata.json` records
`peak_rss_gb` and `timings_seconds`, so tighten the estimate from real numbers.

With `--jobs > 1` each movie runs in a fresh process, its output goes to `<output>/logs/<movie>.log`,
and one progress line per finished movie is printed. A failed movie does not stop the others.

## Crash safety

A movie is written to `<movie>.partial` and renamed to `<movie>` only when complete. A killed or
pre-empted job leaves at most a `.partial` folder, which the next run removes. A folder is complete
if and only if it has `metadata.json`.

## On a cluster (SLURM job array)

One array task per movie, or fewer tasks each taking a share. Inside an array each task runs one
movie at a time and takes its share of the (sorted) movie list from `SLURM_ARRAY_TASK_ID` and
`SLURM_ARRAY_TASK_COUNT`; threads come from `SLURM_CPUS_PER_TASK`.

```bash
#!/bin/bash
#SBATCH --job-name=traintracks
#SBATCH --array=0-31            # 32 movies -> 32 tasks (0-7 gives 8 tasks of 4 movies each)
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
conda activate TrainTracks      # or: module load ...; source venv/bin/activate
export TRAINTRACKS_RAW_ROOT=/path/to/raw_data
export TRAINTRACKS_RESULTS_ROOT=/path/to/results
python scripts/batch_pipeline.py --skip-existing
# afterwards, once: python scripts/batch_pipeline.py --summarize-only
```

Without SLURM, shard by hand: `--task-index 0 --num-tasks 4` (and 1, 2, 3 on other machines).
Each shard writes `batch_summary.shardIofN.csv`; `--summarize-only` rebuilds the combined
`batch_summary.csv` from all `metadata.json` files. The example script has not been run on a
cluster yet; the SLURM variables used are standard but check them against your site's setup.

## Checking that a fresh run reproduces an archived one

```bash
python scripts/compare_runs.py <old_root> <new_root>              # all movies
python scripts/compare_runs.py <old_root> <new_root> --movie sg100_Well5_1018
```

Compares blobs, flow (bit for bit), tracks and clean tracks (by `track_uid`), flow scores (by
`track_uid`) and the counts/parameters in `metadata.json`.
