# Copyright (C) Mitsubishi Electric Research Labs (MERL) 2023
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import pickle
import subprocess
from contextlib import contextmanager
from pathlib import Path
from pprint import pformat
from time import time
from typing import List, Optional

import numpy as np
import torch
from loguru import logger
from pytorch_lightning import seed_everything
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tqdm import tqdm

PROJECT_PATH = Path(__file__).resolve().parent.parent
VENV_PATH = PROJECT_PATH / "venv"


@contextmanager
def add_start_stop_flags(folder):
    """Adds:
    - "STARTED.txt", so we know that jobs launched successfully.
    - on success, "FINISHED.txt" containing stop and elapsed times, so we know the run succeeded.
    - on error, "ERROR.txt" containing start, stop, elasped, and exception.
    """
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)

    t0 = time()
    (folder / "STARTED.txt").touch()
    try:
        yield
        t1 = time()
        with open(folder / "FINISHED.txt", "w") as f:
            f.write(f"Elapsed: {t1 - t0:.2f}s")
    except Exception as e:
        t1 = time()
        with open(folder / "ERROR.txt", "w") as f:
            f.write(f"Elapsed: {t1 - t0:.2f}s")
        raise e


def float01(f):
    f = float(f)
    assert 0 < f < 1, f"Value should be in [0, 1]: {f}"
    return f


def str2bool(s):
    return s.lower() in ["yes", "true", "y"]


def get_git_hash():
    """Get short git hash, with "+" suffix if local files modified"""
    # Need to update index, otherwise can give incorrect results
    subprocess.run(["git", "update-index", "-q", "--refresh"])
    h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip().decode("utf-8")

    # Add '+' suffix if local files are modified
    exitcode, _ = subprocess.getstatusoutput("git diff-index --quiet HEAD")
    if exitcode != 0:
        h += "+"
    return "git" + h


def class_inverse_freqs(labels: torch.Tensor) -> torch.Tensor:
    counts = labels.bincount()
    return counts.sum() / counts


def parse_tensorboard_files(
    path: Path,
    pickle_name: str = "parsed_tensorboard.pkl",
    force_reprocess: bool = False,
    glob="events.out.tfevents*",
    truncate: Optional[int] = None,
):
    """Given a folder containing one or more tensorboard events files, obtain a dictionary with contents of all
    tensorboard scalar graphs (from `SummaryWriter.add_scalar(...)`).

    Args:
        path (Path): folder with one or more tensorboard events files
        pickle_name (str): name of file to save processed results
        force_reprocess (bool): if True, ignores existing preprocessed results. (set to True after creating new runs)

    Returns:
        result = {
            "metadata": {...},
            "runs": {
                <run_name1>: {
                    "best_ckpt": <path_to_best_ckpt>,
                    "last_ckpt": <path_to_last_ckpt>,
                    <tag_name1>: {
                        "steps": <x_axis_values>,
                        "values": <y_axis_values>,
                    },
                    <tag_name2>: {
                        "steps": <x_axis_values>,
                        "values": <y_axis_values>,
                    },
                    ...
                },
                <run_name2>: ...,
                ...
            }
        }
    """
    result_path = path / pickle_name
    logger.info(f"Extract tensorboard event files from: {path} using glob: {glob} into file: {result_path}")
    if not path.exists():
        raise ValueError(f"Path does not exist: {path}")

    if not force_reprocess and result_path.exists():
        logger.info("Using previously processed results")
        with open(result_path, "rb") as f:
            return pickle.load(f)

    events_files = list(path.rglob(glob))
    if truncate is not None:
        events_files = events_files[:truncate]
    logger.info(f"Processing {len(events_files)} events files: \n{pformat(events_files)}")

    result = {"metadata": {"source_path": str(path)}, "runs": {}}
    for f in tqdm(events_files, desc="Files", position=0, leave=True):
        # One file, like .../results/run1/arg1/events.out.tfevents...0
        name = str(f.parent)

        # Extract best and last checkpoint files
        # NOTE - fragile to directory structure - expects checkpoints live alongside events file
        best_ckpt = str(list(f.parent.rglob("*checkpoint.best.pt"))[0])
        last_ckpt = str(list(f.parent.rglob("*checkpoint.pt"))[0])

        # Extract contents of graphs
        result["runs"][name] = {"best_ckpt": best_ckpt, "last_ckpt": last_ckpt}
        summary = EventAccumulator(str(f)).Reload()
        for t in tqdm(summary.Tags()["scalars"], desc="graphs", position=1, leave=False):
            # One tag for a scalar graph, like "Train/loss"
            events = summary.Scalars(t)
            steps = [e.step for e in events]  # x-axis values
            values = [e.value for e in events]  # y-axis values
            result["runs"][name][t] = {"steps": steps, "values": values}

    with open(result_path, "wb") as f:
        pickle.dump(result, f)
    return result


def shuffle(*, seed: int, arrays: List[np.ndarray]):
    seed_everything(seed)
    first_len = len(arrays[0])
    assert all(len(a) == first_len for a in arrays)
    perm = np.random.permutation(first_len)
    return [a[perm] for a in arrays]
