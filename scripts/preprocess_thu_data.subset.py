# Copyright (C) Mitsubishi Electric Research Labs (MERL) 2023
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Subset to 24 train / 4 val / 4 test subjects per fold, and reduce class imbalance to ~1:10"""
import numpy as np
from thu_rsvp_dataset import THU_RSVP_Dataset, get_default_transform

from src.utils import PROJECT_PATH, shuffle

transform = get_default_transform(
    sample_rate_hz=250,  # Sample rate of original dataset
    notch_freq_hz=50,  # AC line frequency in China
    notch_quality_factor=30,
    bandpass_low=1,  # Low frequency cutoff
    bandpass_high=20,  # High frequency cutoff
    bandpass_order=5,  # Order of Butterworth filter
    downsample_factor=2,  # Effective frequency 125 Hz (ample for 1-20 bandpass, roughly square shaped inputs)
)

# Load dataset
data_dir = PROJECT_PATH / "datasets" / "preprocessed" / "thu_rsvp" / "subset"
data_dir.mkdir(exist_ok=True, parents=True)
thu_rsvp = THU_RSVP_Dataset(
    dir=data_dir.parent,
    trial_duration_ms=500,
    transform=transform,
    download=True,
    verify_sha256=False,
    verbose=True,
    force_extract=False,  # NOTE - set this to true after changing transforms
)
x, y, su, se = thu_rsvp.get_data()
if np.unique(su).min() == 1:
    su -= 1  # NOTE - subject ids in [1, 64], want [0, 63]
print(f"{np.unique(y)=}, {np.unique(su)=}, {np.unique(se)=}")
print(f"{x.dtype=}, {y.dtype=}, {su.dtype=}, {se.dtype=}")


def filter_by_vals(arr_to_filter, desired_vals_list):
    ind = np.isin(arr_to_filter, desired_vals_list)
    return x[ind], y[ind], su[ind], se[ind]


# Divide the data into pre-set train/val/test splits based on subject id
# 64 subjects total. In each fold, use 24 subj train, 4 subj val, 4 subj test.
fold_ids = []
S_total = 64
N_val_test = 4
N_train = 24
N_FOLDS = 10
for fold in range(N_FOLDS):
    val = np.arange(fold * N_val_test, (fold + 1) * N_val_test) % S_total
    test = np.arange((fold + 1) * N_val_test, (fold + 2) * N_val_test) % S_total
    train = np.arange((fold + 2) * N_val_test, (fold + 2) * N_val_test + N_train) % S_total
    assert len(np.intersect1d(val, test)) == 0
    assert len(np.intersect1d(val, train)) == 0
    assert len(np.intersect1d(train, test)) == 0

    fold_ids.append((train, val, test))


def reduce_class_imbalance_one_subj(_x, _y, _su, _se, factor: int):
    """Given N examples of minority class, keeps first factor*N examples of majority class."""
    counts = np.bincount(_y)
    n_minority = counts.min()
    n_majority_desired = factor * n_minority

    majority_class = counts.argmax()
    minority_class = counts.argmin()

    majority_class_idx = np.where(_y == majority_class)[0]
    majority_keep_idx = majority_class_idx[:n_majority_desired]

    minority_keep_idx = np.where(_y == minority_class)[0]

    all_keep_idx = np.union1d(minority_keep_idx, majority_keep_idx)
    assert len(all_keep_idx) == (len(minority_keep_idx) + len(majority_keep_idx))

    return _x[all_keep_idx], _y[all_keep_idx], _su[all_keep_idx], _se[all_keep_idx]


def reduce_class_imbalance(_x, _y, _su, _se, factor=10):
    result_x, result_y, result_su, result_se = [], [], [], []
    for one_subj_id in np.unique(_su):
        idx = np.where(_su == one_subj_id)[0]
        tmp_x, tmp_y, tmp_su, tmp_se = reduce_class_imbalance_one_subj(
            _x[idx], _y[idx], _su[idx], _se[idx], factor=factor
        )
        result_x.append(tmp_x)
        result_y.append(tmp_y)
        result_su.append(tmp_su)
        result_se.append(tmp_se)
    return np.concatenate(result_x), np.concatenate(result_y), np.concatenate(result_su), np.concatenate(result_se)


for fold, (su_train, su_val, su_test) in enumerate(fold_ids):
    print(f"Get fold {fold}. {su_train=}, {su_val=}, {su_test=}")
    x_train, y_train, su_train, se_train = filter_by_vals(su, su_train)
    x_train, y_train, su_train, se_train = reduce_class_imbalance(x_train, y_train, su_train, se_train)
    x_train, y_train, su_train, se_train = shuffle(seed=0, arrays=[x_train, y_train, su_train, se_train])

    x_val, y_val, su_val, se_val = filter_by_vals(su, su_val)
    x_val, y_val, su_val, se_val = reduce_class_imbalance(x_val, y_val, su_val, se_val)
    x_val, y_val, su_val, se_val = shuffle(seed=0, arrays=[x_val, y_val, su_val, se_val])

    x_test, y_test, su_test, se_test = filter_by_vals(su, su_test)
    x_test, y_test, su_test, se_test = reduce_class_imbalance(x_test, y_test, su_test, se_test)
    x_test, y_test, su_test, se_test = shuffle(seed=0, arrays=[x_test, y_test, su_test, se_test])

    if not len(np.unique(su_val)) == len(np.unique(su_test)) == N_val_test:
        breakpoint()

    if not len(np.unique(su_train)) == N_train:
        breakpoint()

    assert len(np.intersect1d(np.unique(su_train), np.unique(su_val))) == 0
    assert len(np.intersect1d(np.unique(su_train), np.unique(su_test))) == 0
    assert len(np.intersect1d(np.unique(su_val), np.unique(su_test))) == 0

    folder = data_dir / f"fold{fold}"
    folder.mkdir(exist_ok=True, parents=True)
    print(f"Save to {folder=}")
    print("Save train...")
    np.save(folder / "x_train.npy", x_train)
    np.save(folder / "y_train.npy", y_train)
    np.save(folder / "su_train.npy", su_train)
    np.save(folder / "se_train.npy", se_train)

    print("Save val...")
    np.save(folder / "x_val.npy", x_val)
    np.save(folder / "y_val.npy", y_val)
    np.save(folder / "su_val.npy", su_val)
    np.save(folder / "se_val.npy", se_val)

    print("Save test...")
    np.save(folder / "x_test.npy", x_test)
    np.save(folder / "y_test.npy", y_test)
    np.save(folder / "su_test.npy", su_test)
    np.save(folder / "se_test.npy", se_test)
