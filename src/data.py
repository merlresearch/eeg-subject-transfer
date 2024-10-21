# Copyright (C) Mitsubishi Electric Research Labs (MERL) 2023
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import random
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
from loguru import logger
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset

from src.utils import PROJECT_PATH, class_inverse_freqs, str2bool

DATA_DIR = PROJECT_PATH / "datasets" / "preprocessed"


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_contiguous(y: torch.Tensor, offset: int = 0):
    tmp = torch.clone(y)
    for idx, val in enumerate(y.unique()):
        tmp[y == val] = idx + offset
    return tmp


class BaseData(ABC):
    @property
    @abstractmethod
    def n_subjects(self): ...

    @property
    @abstractmethod
    def n_classes(self): ...

    @property
    @abstractmethod
    def input_chans(self): ...

    @property
    @abstractmethod
    def input_time_length(self): ...

    def __init__(self, x: np.ndarray, y: np.ndarray, s: np.ndarray, fold_idx: int, n_subjects: int):
        """Return train, val, and test split for the desired fold_idx"""
        if fold_idx >= n_subjects:
            raise ValueError(f"Invalid fold_idx: {fold_idx}")

        x = torch.tensor(x).float()
        y = torch.tensor(y).long().squeeze()
        s = torch.tensor(s).long()

        # Make labels and subj_id contiguous ranges
        y = make_contiguous(y)
        s = make_contiguous(s)

        val_subj = np.array(fold_idx)
        test_subj = np.array((fold_idx + 1) % n_subjects)
        train_subj = np.setdiff1d(np.setdiff1d(np.arange(n_subjects), val_subj), test_subj)

        is_train = np.isin(s, train_subj)
        is_val = np.isin(s, val_subj)
        is_test = np.isin(s, test_subj)

        self.train_set = TensorDataset(x[is_train], y[is_train], s[is_train])
        self.val_set = TensorDataset(x[is_val], y[is_val], s[is_val])
        self.test_set = TensorDataset(x[is_test], y[is_test], s[is_test])
        self.class_weights = class_inverse_freqs(y[is_train])


def compute_joint_nuisance_label(su, se):
    """Compute a nuisance variable that indexes (subj, sess).
    We do this easily by treating su as the 10's digit and se as the one's,
    and using make_contiguous after."""
    assert se.max() < 10
    res = torch.zeros_like(su)
    for u in su.unique():
        su_idx = su == u
        for e in se[su_idx].unique():
            idx = su_idx & (se == e)
            res[idx] = 10 * u + e
    res = make_contiguous(res, 0)
    return res


class THU(BaseData):
    n_subjects = 64
    n_classes = 2
    input_chans = 62
    input_time_length = 63

    def __init__(self, fold_idx: int, use_val_set=True, normalize=False, use_joint_nuisance=True):
        path = DATA_DIR / "thu_rsvp" / "subset" / f"fold{fold_idx}"
        x_train = torch.from_numpy(np.load(path / "x_train.npy"))
        y_train = torch.from_numpy(np.load(path / "y_train.npy"))
        su_train = torch.from_numpy(np.load(path / "su_train.npy"))  # subject ids
        se_train = torch.from_numpy(np.load(path / "se_train.npy"))  # session_ids

        x_val = torch.from_numpy(np.load(path / "x_val.npy"))
        y_val = torch.from_numpy(np.load(path / "y_val.npy"))
        su_val = torch.from_numpy(np.load(path / "su_val.npy"))
        se_val = torch.from_numpy(np.load(path / "se_val.npy"))

        x_test = torch.from_numpy(np.load(path / "x_test.npy"))
        y_test = torch.from_numpy(np.load(path / "y_test.npy"))
        su_test = torch.from_numpy(np.load(path / "su_test.npy"))
        se_test = torch.from_numpy(np.load(path / "se_test.npy"))

        if not use_val_set:
            x_train = torch.cat((x_train, x_val))
            y_train = torch.cat((y_train, y_val))
            su_train = torch.cat((su_train, su_val))
            se_train = torch.cat((se_train, se_val))
            del x_val, y_val, su_val, se_val

        self.class_weights = class_inverse_freqs(y_train)

        if normalize:
            # NOTE - affects magnitude of augmentations
            train_means = x_train.mean(dim=(0, 2), keepdims=True)
            train_stds = x_train.std(dim=(0, 2), keepdims=True)

            x_train -= train_means
            x_train /= train_stds

            if use_val_set:
                x_val -= train_means
                x_val /= train_stds

            x_test -= train_means
            x_test /= train_stds

        if use_joint_nuisance:
            # a single index that enumerates both subject and session
            su_train = compute_joint_nuisance_label(su_train, se_train)
            if use_val_set:
                su_val = compute_joint_nuisance_label(su_val, se_val)
            su_test = compute_joint_nuisance_label(su_test, se_test)

        self.n_train_subj = su_train.unique().numel()
        self.n_val_subj = su_val.unique().numel() if use_val_set else 0
        self.n_test_subj = su_test.unique().numel()
        self.n_subj_total = self.n_train_subj + self.n_val_subj + self.n_test_subj

        su_train = make_contiguous(su_train, 0)
        if use_val_set:
            su_val = make_contiguous(su_val, self.n_train_subj)
        su_test = make_contiguous(su_test, self.n_train_subj + self.n_val_subj)

        self.train_set = TensorDataset(x_train, y_train, su_train)
        self.val_set = TensorDataset(x_val, y_val, su_val) if use_val_set else None
        self.test_set = TensorDataset(x_test, y_test, su_test)


class THUDataModule(LightningDataModule):
    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("THUDataModule")
        parser.add_argument("--fold", type=int, required=True)
        parser.add_argument("--batch_size", type=int, default=1024)
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--use_joint_nuisance", type=str2bool, default=True)
        parser.add_argument("--use_val_set", type=str2bool, default=True, help="if false, val subj included in train")
        return parent_parser

    def __init__(
        self,
        fold: int,
        batch_size: int,
        num_workers: int,
        use_joint_nuisance: bool,
        use_val_set: bool,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.fold = fold
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.use_joint_nuisance = use_joint_nuisance
        self.use_val_set = use_val_set

    def setup(self, stage: Optional[str] = None):
        self.thu = THU(self.fold, use_val_set=self.use_val_set, use_joint_nuisance=self.use_joint_nuisance)
        self.num_classes = THU.n_classes
        self.class_weights = tuple(self.thu.class_weights.tolist())
        self.input_shape = tuple(self.thu.train_set.tensors[0][0].shape)
        self.n_train_subj = self.thu.n_train_subj
        logger.info("Setup datamodule:")
        logger.info(f"{self.num_classes=}")
        logger.info(f"{self.class_weights=}")
        logger.info(f"{self.input_shape=}")
        logger.info(f"{self.n_train_subj=}")

    def one_loader(self, dset, shuffle=True):
        return DataLoader(
            dset,
            batch_size=self.batch_size,
            pin_memory=True,
            worker_init_fn=seed_worker,
            num_workers=self.num_workers,
            shuffle=shuffle,
            generator=torch.Generator().manual_seed(self.seed),
        )

    def train_dataloader(self):
        return self.one_loader(self.thu.train_set)

    def val_dataloader(self):
        if self.use_val_set:
            return self.one_loader(self.thu.val_set, False)
        else:
            raise ValueError(f"Tried to fetch val dataloader, {self.use_val_set=}")

    def test_dataloader(self):
        return self.one_loader(self.thu.test_set, False)
