<!--
Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Stabilizing Subject Transfer in EEG Classification with Divergence Estimation

Code for "*Stabilizing Subject Transfer in EEG Classification with Divergence Estimation*" by Niklas Smedemark-Margulies, Ye Wang, Toshiaki Koike-Akino, Jing Liu, Kieran Parsons, Yunus Bicer, and Deniz Erdogmus.

In this work, we provide two new censoring methods based on density ratio estimation and Wasserstein-1 divergence.
We neural network critic functions to estimate a regularization penalty based on one of these methods, and evaluate their performance against an adversarial classifier baseline using a large EEG benchmark dataset.

## Setup
### 1. Setup project

Setup project using:
```shell
make
```
The default Makefile target will setup the project (if not already setup) and format/lint all files.
Note that it uses `python3.8`; if this is not available or python from a specific path should be used, change `BASE_PYTHON` to a suitable path (e.g. `/usr/bin/python3.8`).
The project should work with roughly python3.7 or newer.

Makefile targets:
- `make setup` - create virtualenv, install python packages, and install pre-commit hooks
- `make lint` - format and lint all files, using the same rules that are part of `pre-commit`
- `make test` - there are unit tests for a subset of code, just some sanity checks.
    Not all tests pass, and some tests expect that data has been downloaded
- `make destroy-setup` - delete the python virtual environment in order to create it again from scratch

Use `source venv/bin/activate` for all commands below to activate virtual environment.

### 2. Prepare data

Run pre-processing script using:
```shell
python scripts/preprocess_thu_data.subset.py
```

The preprocessing script does the following steps:
1. Downloads raw files for THU RSVP benchmark dataset
2. Applies signal filtering/pre-processing steps (notch filter, bandpass filter, downsample) and extract labeled trials using:
    ```python
    transform = get_default_transform(...)
    thu_rsvp = THU_RSVP_Dataset(..., transform=transform)
    ```
    NOTE - `force_extract=False` will just re-use previously extracted trials (for speed).
    In order to adjust the signal filtering stage, change the transform and use `force_extract=True`:
    ```python
    new_transform = get_default_transform(...)  # Suppose you make changes here...
    thu_rsvp = THU_RSVP_Dataset(...,
        transform=new_transform,
        force_extract=True,                     # ...Then you must set this flag
    )
    ```
3. Divides into several train/val/test folds and saves as `*.npy` files for fast loading during training runs
    Note that [src/data.py]() has a hard-coded path for the location of these `*.npy` files.
    (Currently `path = DATA_DIR / "thu_rsvp" / "subset" / f"fold{fold_idx}"`)
    If using different train/val/test splits be sure to update this line in [src/data.py]().


### 3. Run experiments

The first experiment uses the final checkpoint for testing:
```shell
python scripts/run_overfitting_experiment.long.py
```

The second experiment uses the best val checkpoint for testing:
```shell
python scripts/run_censoring_experiment.py
```

NOTE - these grids are very large and slurm queue can only hold ~10K jobs.
- Set `DRY_RUN=True` at top of script to see jobs that will be run and try a few manually
- Comment out subsets of params within script as needed.
- Try not to run baseline multiple times (it wastes some time, though it should be fine; can select only final baseline run when loading results, but wastes some time)

Results will go into `results/<FOLDER>` with different name depending on script.
- `results/censoring__last__100epoch` (we used this folder name for the main results in the paper)
- `results/censoring` (we used this folder name for results that combine censoring with early stopping in paper appendix)


### 4. Make plots

Plots are created in two steps, to allow for more easy development.
First, scrape results into a pickle file:
```shell
python scripts/make_plots_1.py --experiment_name <NAME>  # <NAME> is "censoring" or "censoring__last__100epoch"
```

Next create plots from pickle file:
```shell
python scripts/make_plots_2.py --experiment_name <NAME>  # <NAME> is "censoring" or "censoring__last__100epoch"
```

## Notes about Dataset

As mentioned above, since the full THU dataset is very large, it is pre-processed into individual numpy files for train/val/test so that data can be quickly loaded during a single training run.

The preprocessing is done using `scripts/preprocess_thu_data.subset.py`
    - Data is split by subjects: 24 train subjects / 4 val / 4 test.
    - Data is also down-sampled so that the class label distribution 1 target : 10 non-target.
    (The original class ratio is ~1 target : 60 non-target)

Other dataset considerations:
    - Using all available subjects (54 train / 5 val / 5 test), and no class downsampling was problematic. Training was too slow for very large experiment grids (~1 hour per run), with occasional OOM issues.
    - Class downsampling seems to be important (the raw class ratio is ~1 target : 60 non-target). With excessively imbalanced classes, the potential benefit of regularization is obscured by the task difficulty.

## PDF

To read our paper, see: https://arxiv.org/pdf/2310.08762.pdf

## Citation

If you use the software, please cite the following:

```BibTeX
@article{smedemark2023stabilizing,
    title={Stabilizing Subject Transfer in EEG Classification with Divergence Estimation},
    author={
        Smedemark-Margulies, Niklas and
        Wang, Ye and
        Koike-Akino, Toshiaki and
        Liu, Jing and
        Parsons, Kieran and
        Bicer, Yunus and
        Erdogmus, Deniz
    },
    journal={arXiv preprint arXiv:2310.08762},
    year={2023}
}
```

## Contact

Ye Wang <yewang@merl.com>

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for our policy on contributions.

## License

Released under `AGPL-3.0-or-later` license, as found in the [LICENSE.md](LICENSE.md) file.

All files:

```
Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL).

SPDX-License-Identifier: AGPL-3.0-or-later
```
