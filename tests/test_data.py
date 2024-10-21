# Copyright (C) Mitsubishi Electric Research Labs (MERL) 2023
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import numpy as np
import torch

from src.data import THU, compute_joint_nuisance_label, make_contiguous


def test_THU():
    thu = THU(fold_idx=0)
    train, val, test = thu.train_set, thu.val_set, thu.test_set
    for d in [train, val, test]:
        assert len(d.tensors) == 3, "should have x, y, s"

    # Check data
    x_train = train.tensors[0]
    assert x_train.shape[1:] == (THU.input_chans, THU.input_time_length), "train data must match specified shape"
    x_val = val.tensors[0]
    assert x_val.shape[1:] == (THU.input_chans, THU.input_time_length), "val data must match specified shape"
    x_test = test.tensors[0]
    assert x_test.shape[1:] == (THU.input_chans, THU.input_time_length), "test data must match specified shape"

    # Check labels
    y_train = train.tensors[1]
    assert np.all(np.unique(y_train) == np.arange(THU.n_classes)), "train labels must be 0..n_classes"
    y_val = val.tensors[1]
    assert np.all(np.unique(y_val) == np.arange(THU.n_classes)), "val labels must be 0..n_classes"
    y_test = test.tensors[1]
    assert np.all(np.unique(y_test) == np.arange(THU.n_classes)), "test labels must be 0..n_classes"

    # Check subj
    s_train = np.unique(train.tensors[2])
    s_val = np.unique(val.tensors[2])
    s_test = np.unique(test.tensors[2])
    assert len(np.intersect1d(s_train, s_val)) == 0, "train and val should be disjoint"
    assert len(np.intersect1d(s_train, s_test)) == 0, "train and test should be disjoint"
    assert len(np.intersect1d(s_val, s_test)) == 0, "val and test should be disjoint"


def test_make_contiguous():
    y1 = torch.tensor([0, 0, 2, 2, 3, 3, 5, 5])
    y2 = make_contiguous(y1)
    assert torch.all(y2.eq(torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])))

    y1 = torch.tensor([0, 0, 5, 5, 2, 2, 3, 3])
    y2 = make_contiguous(y1)
    assert torch.all(y2.eq(torch.tensor([0, 0, 3, 3, 1, 1, 2, 2])))


def test_compute_joint_nuisance_label():
    su = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    se = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])

    res = compute_joint_nuisance_label(su, se)
    assert torch.all(res.eq(torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])))

    su = torch.tensor([0, 0, 0, 0, 5, 5, 5, 5, 7, 7, 7, 7])
    se = torch.tensor([0, 0, 3, 4, 2, 2, 2, 2, 0, 3, 1, 1])

    res = compute_joint_nuisance_label(su, se)
    assert torch.all(res.eq(torch.tensor([0, 0, 1, 2, 3, 3, 3, 3, 4, 6, 5, 5])))
