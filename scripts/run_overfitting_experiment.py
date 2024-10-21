# Copyright (C) Mitsubishi Electric Research Labs (MERL) 2023
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Examine effect of censoring hyperparams"""

import subprocess
from itertools import product
from textwrap import dedent

from src.utils import PROJECT_PATH

DRY_RUN = False

script = PROJECT_PATH / "scripts" / "censoring_entrypoint.py"
results_dir = "censoring__last"
batch_size = 1024
latent_dim = 128
lr = 1e-4
censor_hidden_layers = 3
proj_hidden_layers = 3
val_check_interval = 1.0
use_val_set = False
use_joint_nuisance = True
epochs = 30

JOB_COUNT = 0


def run_one(
    name_prefix: str,
    fold: int,
    feature_type: str,
    censor_type: str,
    censor_mode: str,
    censor_weight: float,
    censor_steps: int,
    seed: int,
):
    # Add "/" to create subdirectories
    name_prefix = f"fold={fold}/seed={seed}/{name_prefix}"
    name_parts = [
        name_prefix,
        f"type={censor_type}",
        f"mode={censor_mode}",
        f"weight={censor_weight}",
        f"steps={censor_steps}",
        f"feature_type={feature_type}",
    ]
    name = "__".join(name_parts)

    cmd = dedent(
        f"""
        slurmrunyes {script}
         --censor_type {censor_type}
         --censor_mode {censor_mode}
         --name {name}
         --fold {fold}
         --accelerator auto
         --deterministic True
         --enable_progress_bar False
         --main_lr {lr}
         --censor_lr {lr}
         --main_lr_decay 1.0
         --censor_lr_decay 1.0
         --latent_dim {latent_dim}
         --censor_weight {censor_weight}
         --censor_steps_per_main_step {censor_steps}
         --feature_type {feature_type}
         --use_joint_nuisance {use_joint_nuisance}
         --censor_hidden_layers {censor_hidden_layers}
         --proj_hidden_layers {proj_hidden_layers}
         --batch_size {batch_size}
         --num_workers 5
         --max_epochs {epochs}
         --val_check_interval {val_check_interval}
         --use_val_set {use_val_set}
         --seed {seed}
         --results_dir {results_dir}
        """
    ).replace("\n", "")

    if DRY_RUN:
        print()
        print(cmd)
    else:
        subprocess.run(cmd, shell=True, check=True)
    global JOB_COUNT
    JOB_COUNT += 1


if DRY_RUN:
    print("DRY RUN: print jobs only")


for fold, seed in product(range(10), range(10)):
    run_one(
        name_prefix="baseline",
        fold=fold,
        feature_type="direct",  # projection not used
        censor_type="wyner",  # doesn't matter
        censor_mode="marginal",  # doesn't matter
        censor_weight=0.0,
        censor_steps=0,  # avoid wasting time
        seed=seed,
    )

    for censor_mode, feature_type, censor_steps, censor_type, censor_weight in product(
        ["marginal", "conditional", "complementary"],
        ["direct", "projected"],
        [1],
        ["wyner", "wasserstein", "adv"],
        [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
    ):
        run_one(
            name_prefix="censored",
            fold=fold,
            feature_type=feature_type,
            censor_type=censor_type,
            censor_mode=censor_mode,
            censor_weight=censor_weight,
            censor_steps=censor_steps,
            seed=seed,
        )

print(f"{JOB_COUNT} total jobs")
