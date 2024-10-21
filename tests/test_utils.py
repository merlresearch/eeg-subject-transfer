# Copyright (C) Mitsubishi Electric Research Labs (MERL) 2023
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import time
from multiprocessing import Process

from src.utils import add_start_stop_flags


def test_start_stop_flags_success(tmp_path):
    @add_start_stop_flags(tmp_path)
    def worker_fn():
        time.sleep(0.1)

    p = Process(target=worker_fn)
    p.start()
    p.join()
    assert (tmp_path / "STARTED.txt").exists()
    assert (tmp_path / "FINISHED.txt").exists()
    assert not (tmp_path / "ERROR.txt").exists()


def test_start_stop_flags_failure(tmp_path):
    @add_start_stop_flags(tmp_path)
    def worker_fn():
        time.sleep(0.1)
        raise ValueError()

    p = Process(target=worker_fn)
    p.start()
    p.join()
    assert (tmp_path / "STARTED.txt").exists()
    assert not (tmp_path / "FINISHED.txt").exists()
    assert (tmp_path / "ERROR.txt").exists()
