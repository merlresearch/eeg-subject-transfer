# Copyright (C) Mitsubishi Electric Research Labs (MERL) 2023
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
import pickle
import re
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils import PROJECT_PATH

layout_template = "simple_white"
colorscale = "viridis"  # try: "thermal", "cividis", https://plotly.com/python/builtin-colorscales/


def read_one_csv(metrics_csv: Path, which_ckpt="best"):
    if which_ckpt not in ["best", "last"]:
        raise ValueError()

    df = pd.read_csv(metrics_csv)

    checkpoints = metrics_csv.parent / "checkpoints"
    ckpts = list(checkpoints.glob(f"{which_ckpt}__epoch=*"))
    if len(ckpts) > 1:
        # If baseline was accidentally rerun, can have duplicates
        # Prefer the one whose timestamp is closest to metrics.csv, so they definitely match
        assert "baseline" in str(metrics_csv)
        metrics_time = metrics_csv.stat().st_mtime
        closest = min(ckpts, key=lambda c: abs(c.stat().st_mtime - metrics_time))
        ckpt_name = closest.name
    else:
        ckpt_name = ckpts[0].name
    match = re.match(rf"^{which_ckpt}__epoch=\d+-step=(\d+)-", ckpt_name)
    assert match is not None
    step = int(match.group(1))

    # Include train acc from 50 batches, not just a single batch
    n_train_batches_measured = 50
    usable_train_accs = df[["step", "train/acc", "train/bal_acc"]].dropna()
    best_train_idx = usable_train_accs.index.get_indexer([step], method="nearest")
    start_idx = int(best_train_idx - n_train_batches_measured)
    end_idx = int(best_train_idx)
    best_train_bal_acc = np.mean(usable_train_accs.iloc[start_idx:end_idx]["train/bal_acc"].values)
    best_train_acc = np.mean(usable_train_accs.iloc[start_idx:end_idx]["train/acc"].values)

    best_test_bal_acc = df["test/bal_acc"].dropna().values[-1]
    best_test_acc = df["test/acc"].dropna().values[-1]

    return {
        "train": {"acc": best_train_acc, "bal_acc": best_train_bal_acc},
        "test": {"acc": best_test_acc, "bal_acc": best_test_bal_acc},
    }


def read_all_csv(results_dir, which_ckpt):
    train_records = defaultdict(list)
    test_records = defaultdict(list)

    df_records = []
    censored_keys = []
    for fold, seed in tqdm(list(product(range(10), range(10))), desc="Loading"):
        # load baseline
        short_name = "baseline"
        full_name = f"fold={fold}/seed={seed}/baseline"
        matching_dirs = list(results_dir.glob(full_name + "*"))
        assert len(matching_dirs) == 1

        vals = read_one_csv(matching_dirs[0] / "metrics.csv", which_ckpt)
        train_records[short_name].append(vals["train"])
        test_records[short_name].append(vals["test"])
        for split in ["train", "test"]:
            df_records.append(
                {
                    "split": split,
                    "fold": fold,
                    "seed": seed,
                    "is_censored": False,
                    "method": None,
                    "mode": None,
                    "steps": None,
                    "feature_type": None,
                    "weight": None,
                    "bal_acc": vals[split]["bal_acc"],
                    "acc": vals[split]["acc"],
                }
            )
        df_records.append(
            {
                "split": "train_minus_test",
                "fold": fold,
                "seed": seed,
                "is_censored": False,
                "method": None,
                "mode": None,
                "steps": None,
                "feature_type": None,
                "weight": None,
                "bal_acc": vals["train"]["bal_acc"] - vals["test"]["bal_acc"],
                "acc": vals["train"]["acc"] - vals["test"]["acc"],
            }
        )
        df_records.append(
            {
                "split": "test_over_train",
                "fold": fold,
                "seed": seed,
                "is_censored": False,
                "method": None,
                "mode": None,
                "steps": None,
                "feature_type": None,
                "weight": None,
                "bal_acc": vals["test"]["bal_acc"] / vals["train"]["bal_acc"],
                "acc": vals["test"]["acc"] / vals["train"]["acc"],
            }
        )

        # load censored models
        for censor_mode, feature_type, censor_steps, censor_type, censor_weight in product(
            ["marginal", "conditional", "complementary"],
            ["direct", "projected"],
            [1],
            ["wyner", "wasserstein", "adv"],
            [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 50.0, 100.0],
        ):
            full_name = f"fold={fold}/seed={seed}/censored__type={censor_type}__mode={censor_mode}"
            full_name += f"__weight={censor_weight}__steps={censor_steps}__feature_type={feature_type}"
            # short name does not include fold and seed - we collect across those
            short_name = f"censored__{feature_type=}__{censor_steps=}__{censor_type=}__{censor_weight=}__{censor_mode=}"
            matching_dirs = list(results_dir.glob(full_name))
            if len(matching_dirs) == 0:
                print(f"Missing run: {full_name}")
                continue
            elif len(matching_dirs) > 1:
                print(f"Duplicated run: {full_name}")
                continue

            vals = read_one_csv(matching_dirs[0] / "metrics.csv", which_ckpt)
            train_records[short_name].append(vals["train"])
            test_records[short_name].append(vals["test"])
            for split in ["train", "test"]:
                df_records.append(
                    {
                        "split": split,
                        "fold": fold,
                        "seed": seed,
                        "is_censored": True,
                        "method": censor_type,
                        "mode": censor_mode,
                        "steps": str(censor_steps),
                        "feature_type": feature_type,
                        "weight": str(censor_weight),
                        "bal_acc": vals[split]["bal_acc"],
                        "acc": vals[split]["acc"],
                    }
                )
            df_records.append(
                {
                    "split": "train_minus_test",
                    "fold": fold,
                    "seed": seed,
                    "is_censored": True,
                    "method": censor_type,
                    "mode": censor_mode,
                    "steps": str(censor_steps),
                    "feature_type": feature_type,
                    "weight": str(censor_weight),
                    "bal_acc": vals["train"]["bal_acc"] - vals["test"]["bal_acc"],
                    "acc": vals["train"]["acc"] - vals["test"]["acc"],
                }
            )
            df_records.append(
                {
                    "split": "test_over_train",
                    "fold": fold,
                    "seed": seed,
                    "is_censored": True,
                    "method": censor_type,
                    "mode": censor_mode,
                    "steps": str(censor_steps),
                    "feature_type": feature_type,
                    "weight": str(censor_weight),
                    "bal_acc": vals["test"]["bal_acc"] / vals["train"]["bal_acc"],
                    "acc": vals["test"]["acc"] / vals["train"]["acc"],
                }
            )

            if fold == 0 and seed == 0:
                censored_keys.append(short_name)
    df = pd.DataFrame.from_records(df_records)

    def name_row(row):
        if not row["is_censored"]:
            return "baseline"

        return "_".join([row["mode"], row["feature_type"]])

    df["name"] = df.apply(name_row, axis=1)
    return df, train_records, test_records, censored_keys


def main(experiment_name, which_ckpt, results_dir, figs_dir):
    df, train_records, test_records, censored_keys = read_all_csv(results_dir, which_ckpt)

    with open(figs_dir / f"preprocessed__{experiment_name}.pkl", "wb") as f:
        pickle.dump([df, train_records, test_records, censored_keys], f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", required=True)
    args = parser.parse_args()

    if args.experiment_name == "censoring":
        args.which_ckpt = "best"
    elif args.experiment_name == "censoring__last__100epoch":
        args.which_ckpt = "last"
    else:
        raise ValueError(f"Unknown experiment name: {args.experiment_name}")

    results_dir = PROJECT_PATH / "results" / args.experiment_name
    figs_dir = PROJECT_PATH / "figures" / args.experiment_name
    figs_dir.mkdir(exist_ok=True, parents=True)

    main(args.experiment_name, args.which_ckpt, results_dir, figs_dir)
