# Copyright (C) Mitsubishi Electric Research Labs (MERL) 2023
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pytorch_lightning import seed_everything
from scipy import stats

from src.utils import PROJECT_PATH

layout_template = "simple_white"
colorscale = "viridis"  # try: "thermal", "cividis", https://plotly.com/python/builtin-colorscales/
discrete_colors = px.colors.qualitative.Plotly


def boxplot_test_acc(df: pd.DataFrame, figs_dir):
    baseline_test_accs = df[(df["name"] == "baseline") & (df["split"] == "test")]["bal_acc"]
    n_colors = df["weight"].unique().size
    colors = px.colors.sample_colorscale(colorscale, n_colors)

    replacements = {}
    for mode in ["marginal", "conditional", "complementary"]:
        for feat in ["direct", "projected"]:
            replacements[f"{mode}_{feat}"] = f"{mode}<br>{feat}"
    df = df.replace(replacements)
    assert len(baseline_test_accs) == 100

    for title, method in [
        ("Adversarial Classifier", "adv"),
        ("Density Ratio Estimator", "wyner"),
        ("Wasserstein Critic", "wasserstein"),
    ]:
        subset_df = df[(df["method"] == method) & (df["split"] == "test")]
        fig = make_subplots(rows=1, cols=3, shared_yaxes=True, column_widths=[1, 0.06, 0.06], horizontal_spacing=0.01)

        dx = 9  # pixel width of each box. Found by trial and error; used to position annotations
        for i, (name, name_group) in enumerate(subset_df.groupby("name", sort=False)):
            for j, (weight, weight_group) in enumerate(name_group.groupby("weight", sort=False)):
                # Create box for this weight + censor mode
                kw = dict(
                    boxmean=True,
                    name=f"$\lambda={weight}$",
                    marker_color=colors[j],
                    jitter=0.5,
                    marker_size=3,
                    showlegend=i == 0,
                    offsetgroup=j,
                    boxpoints=False,
                )
                fig.add_trace(go.Box(x=weight_group["name"], y=weight_group["bal_acc"], **kw), row=1, col=1)

                # Add annotation for statistical significance
                statistic, pvalue = stats.ttest_rel(
                    weight_group["bal_acc"], baseline_test_accs, alternative="two-sided"
                )
                x = name
                xshift = dx * (j - 8)  # Manually match xshift to offsetgroup
                y = np.max(weight_group["bal_acc"]) + 0.01  # Place annotation just above top whisker
                if statistic >= 0:
                    text = choose_significance_symbol(pvalue, "-")
                else:
                    text = ""  # Use "." for testing xshift alignment
                # xref="x" means we can use string tick values like "marginal_direct"
                fig.add_annotation(xref="x", x=x, y=y, xshift=xshift, text=text, showarrow=False, row=1, col=1)

        # Add second plot showing overall performance for this method
        # Pool points from best weight of each censor mode
        # For each method, for each censoring mode, find the best weight by mean bal_acc
        best_weight_each_mode = {}
        for name, name_group in subset_df.groupby("name", sort=False):
            vals_this_method = {}
            for weight, weight_group in name_group.groupby("weight", sort=False):
                assert len(weight_group) == 100
                vals_this_method[weight] = np.mean(weight_group["bal_acc"])
            best_weight_this_method = sorted(vals_this_method.items(), key=lambda x: x[1], reverse=True)[0]
            best_weight_each_mode[name] = best_weight_this_method[0]
        x0 = r"$\text{best } \lambda \\ \text{values}$"
        accs = []
        for name, name_group in subset_df.groupby("name", sort=False):
            weight = best_weight_each_mode[name]
            accs.extend(name_group[name_group["weight"] == weight]["bal_acc"].tolist())
        # Show Violin
        kw = dict(
            box_visible=True,
            points=False,
            marker_color="gray",
            marker_size=3,
            offsetgroup=0,
            showlegend=False,
        )
        fig.add_trace(go.Violin(x0=x0, y=accs, **kw), row=1, col=2)

        # Add third plot showing overall performance for this method
        # Pool all points
        x0 = "all"
        accs = subset_df["bal_acc"]
        # Show Violin
        kw = dict(
            box_visible=True,
            points=False,
            marker_color="gray",
            marker_size=3,
            offsetgroup=0,
            showlegend=False,
        )
        fig.add_trace(go.Violin(x0=x0, y=accs, **kw), row=1, col=3)

        lo, med, hi = np.percentile(baseline_test_accs, [25, 50, 75])
        mean = np.mean(baseline_test_accs)
        kw = dict(line_color="black", line_width=2, opacity=0.8)
        fig.add_hline(y=lo, line_dash="solid", **kw)
        fig.add_hline(y=med, line_dash="solid", **kw)
        fig.add_hline(y=hi, line_dash="solid", **kw)
        fig.add_hline(y=mean, line_dash="dot", **kw)
        fig.update_layout(
            boxmode="group",
            template=layout_template,
            yaxis=dict(range=[0.5, 0.82], title="Balanced Test Acc"),
            legend=dict(orientation="h", yanchor="bottom", y=1.00, xanchor="right", x=1),
            boxgap=0.2,
            boxgroupgap=0.05,
            margin=dict(l=0, r=0, t=10, b=0),
            font_size=20,
        )

        fig.write_image(figs_dir / f"boxplot_test_acc__method={method}.jpeg", width=1200 + 200, height=500, scale=3)


def add_pvalue_annotation(fig: go.Figure, x0, x1, y0, y1, symbol):
    """
    arguments:
    fig - figure handle
    x0 - index of first x-axis group
    x1 - index of second x-axis group
    y0, y1 - describes height of bracket annotation in paper units (e.g. [1.01, 1.02])
    symbol - string to put above center of bracket
    """
    # first vertical tick
    fig.add_shape(type="line", xref="x", yref="paper", x0=x0, y0=y0, x1=x0, y1=y1, line=dict(color="black", width=2))
    # horizontal line
    fig.add_shape(type="line", xref="x", yref="paper", x0=x0, y0=y1, x1=x1, y1=y1, line=dict(color="black", width=2))
    # second vertical tick
    fig.add_shape(type="line", xref="x", yref="paper", x0=x1, y0=y1, x1=x1, y1=y0, line=dict(color="black", width=2))
    ## add text at the correct x, y coordinates
    ## for bars, there is a direct mapping from the bar number to 0, 1, 2...
    fig.add_annotation(
        font=dict(color="black", size=14),
        x=(x0 + x1) / 2,
        y=y1 + 0.04,
        showarrow=False,
        text=symbol,
        textangle=0,
        xref="x",
        yref="paper",
    )


def choose_significance_symbol(pvalue, non_significant_symbol="ns"):
    if 0.05 < pvalue:
        return non_significant_symbol
    elif 0.01 < pvalue <= 0.05:
        return "*"
    elif 0.001 < pvalue <= 0.01:
        return "†"
    else:
        return "‡"


def compare_methods_overall(df: pd.DataFrame, figs_dir: Path):
    """High-level comparisons across methods.
    - given optimal value of lambda, wyner gives higher acc than adv
    - same, but looking at generalization ratio (test_acc / train_acc)
    - across choices of lambda / censor mode, wasserstein is more stable (lower variance) than adv
    """
    # For each method, in each mode, select the best lambda
    # Collect these accs to obtain 3 collections
    # Then do a boxplot with these three

    # Also:
    # For each method, from each mode + lambda value, find the average acc.
    # Collect these 6 * 17 means to obtain 3 collections
    # Then do a boxplot with these three and print standard deviations
    best_bal_accs_each_method = {}
    best_test_train_ratios_each_method = {}  # For the same selected lambda, also compute test/train ratio
    train_accs_each_method = {}  # train acc for same selected lambdas
    avgs_across_hparams_each_method = {}
    best_raw_accs_each_method = {}  # raw acc instead of balanced acc

    baseline_test_accs = df[(df["name"] == "baseline") & (df["split"] == "test")]["bal_acc"]
    baseline_train_accs = df[(df["name"] == "baseline") & (df["split"] == "train")]["bal_acc"]
    baseline_raw_accs = df[(df["name"] == "baseline") & (df["split"] == "test")]["acc"]
    assert len(baseline_test_accs) == 100
    assert len(baseline_train_accs) == 100
    assert len(baseline_raw_accs) == 100

    for method in ["adv", "wyner", "wasserstein"]:
        subset_df = df[(df["method"] == method) & (df["split"] == "test")]
        subset_df_train = df[(df["method"] == method) & (df["split"] == "train")]
        avgs_across_hparams_each_method[method] = []

        best_weight_each_mode = {}
        for name, name_group in subset_df.groupby("name", sort=False):
            vals_this_method = {}
            for weight, weight_group in name_group.groupby("weight", sort=False):
                assert len(weight_group) == 100
                this_mean = np.mean(weight_group["bal_acc"])
                vals_this_method[weight] = this_mean
                avgs_across_hparams_each_method[method].append(this_mean)

            best_weight_this_method = sorted(vals_this_method.items(), key=lambda x: x[1], reverse=True)[0]
            best_weight_each_mode[name] = best_weight_this_method[0]

        test_bal_accs_this_method = []
        test_train_ratios_this_method = []
        train_accs_this_method = []
        test_raw_accs_this_method = []
        for name, weight in best_weight_each_mode.items():
            test_bal_accs = subset_df[(subset_df["name"] == name) & (subset_df["weight"] == weight)]["bal_acc"]
            test_raw_accs = subset_df[(subset_df["name"] == name) & (subset_df["weight"] == weight)]["acc"]
            train_bal_accs = subset_df_train[(subset_df_train["name"] == name) & (subset_df_train["weight"] == weight)][
                "bal_acc"
            ]
            test_train_ratios_this_method.extend(test_bal_accs.values / train_bal_accs.values)
            test_bal_accs_this_method.extend(test_bal_accs.values)
            train_accs_this_method.extend(train_bal_accs.values)
            test_raw_accs_this_method.extend(test_raw_accs.values)

        best_bal_accs_each_method[method] = test_bal_accs_this_method
        best_raw_accs_each_method[method] = test_raw_accs_this_method
        best_test_train_ratios_each_method[method] = test_train_ratios_this_method
        train_accs_each_method[method] = train_accs_this_method

    fig1 = go.Figure()  # Test accuracy, where values use best lambda from each censor mode
    fig2 = go.Figure()  # Examine mean acc across all lambda values and all censor modes (variation due to hparams)
    fig3 = go.Figure()  # Test acc / Train acc, using same experiments as fig1
    fig4 = go.Figure()  # Train acc
    fig5 = go.Figure()  # Raw test acc

    method_names = {
        "adv": "Adversarial",
        "wyner": "Density Ratio",
        "wasserstein": "Wasserstein",
    }

    # baseline values of test bal acc
    lo, med, hi = np.percentile(baseline_test_accs, [25, 50, 75])
    mean = np.mean(baseline_test_accs)
    kw = dict(line_color="black", line_width=2, opacity=0.8)
    fig1.add_hline(y=lo, line_dash="solid", **kw)
    fig1.add_hline(y=med, line_dash="solid", **kw)
    fig1.add_hline(y=hi, line_dash="solid", **kw)
    fig1.add_hline(y=mean, line_dash="dot", **kw)

    # in second plot, only mean makes sense; unregularized model is only run in 1 setting (1 lambda/censor mode)
    fig2.add_hline(y=mean, line_dash="dot", **kw)

    # baseline values of test/train ratio
    baseline_test_train_ratios = baseline_test_accs.values / baseline_train_accs.values
    lo, med, hi = np.percentile(baseline_test_train_ratios, [25, 50, 75])
    mean = np.mean(baseline_test_train_ratios)
    kw = dict(line_color="black", line_width=2, opacity=0.8)
    fig3.add_hline(y=lo, line_dash="solid", **kw)
    fig3.add_hline(y=med, line_dash="solid", **kw)
    fig3.add_hline(y=hi, line_dash="solid", **kw)
    fig3.add_hline(y=mean, line_dash="dot", **kw)

    # baseline values of train bal acc
    lo, med, hi = np.percentile(baseline_train_accs, [25, 50, 75])
    mean = np.mean(baseline_train_accs)
    kw = dict(line_color="black", line_width=2, opacity=0.8)
    fig4.add_hline(y=lo, line_dash="solid", **kw)
    fig4.add_hline(y=med, line_dash="solid", **kw)
    fig4.add_hline(y=hi, line_dash="solid", **kw)
    fig4.add_hline(y=mean, line_dash="dot", **kw)

    # baseline values of train bal acc
    lo, med, hi = np.percentile(baseline_raw_accs, [25, 50, 75])
    mean = np.mean(baseline_raw_accs)
    kw = dict(line_color="black", line_width=2, opacity=0.8)
    fig5.add_hline(y=lo, line_dash="solid", **kw)
    fig5.add_hline(y=med, line_dash="solid", **kw)
    fig5.add_hline(y=hi, line_dash="solid", **kw)
    fig5.add_hline(y=mean, line_dash="dot", **kw)

    for i, method in enumerate(["adv", "wyner", "wasserstein"]):
        print("*" * 20)
        print("Method:", method)

        # Update fig1 (best lambdas from each mode)
        accs = best_bal_accs_each_method[method]
        # compare this method to unregularized model
        statistic, pvalue = stats.ttest_ind(accs, baseline_test_accs, alternative="greater", equal_var=False)
        print(f"T-test against unregularized: {statistic=:.3f}, {pvalue=:.3f}")

        fig1.add_trace(
            go.Box(
                x0=method_names[method],
                y=accs,
                name=method,
                boxpoints=False,
                boxmean=True,
                showlegend=False,
                offsetgroup=1,
                marker_color=discrete_colors[i],
            )
        )

        # Update fig2 (means across all lambdas all modes)
        accs = avgs_across_hparams_each_method[method]
        kw = dict(
            name=method,
            showlegend=False,
            boxpoints="all",
            marker_color=discrete_colors[i],
            pointpos=0,
            jitter=0.5,
            line_color="rgba(0,0,0,0)",
            fillcolor="rgba(0,0,0,0)",
            offsetgroup=1,
        )
        fig2.add_trace(go.Box(x0=method_names[method], y=accs, **kw))
        print()

        # Update fig3 (test/train ratios)
        test_train_ratios = best_test_train_ratios_each_method[method]
        fig3.add_trace(
            go.Box(
                x0=method_names[method],
                y=test_train_ratios,
                name=method,
                boxpoints=False,
                boxmean=True,
                showlegend=False,
                offsetgroup=1,
                marker_color=discrete_colors[i],
            )
        )

        # Update fig4 (train accs)
        train_accs = train_accs_each_method[method]
        fig4.add_trace(
            go.Box(
                x0=method_names[method],
                y=train_accs,
                name=method,
                boxpoints=False,
                boxmean=True,
                showlegend=False,
                offsetgroup=1,
                marker_color=discrete_colors[i],
            )
        )

        # Update fig5 (raw test accs)
        test_raw_accs = best_raw_accs_each_method[method]
        fig5.add_trace(
            go.Box(
                x0=method_names[method],
                y=test_raw_accs,
                name=method,
                boxpoints=False,
                boxmean=True,
                showlegend=False,
                offsetgroup=1,
                marker_color=discrete_colors[i],
            )
        )

    kw = dict(
        boxmode="group",
        template=layout_template,
        yaxis=dict(range=[0.5, 0.82], title="Balanced Test Acc"),
        legend=dict(orientation="h", yanchor="bottom", y=1.00, xanchor="right", x=1),
        boxgap=0.2,
        boxgroupgap=0,
        margin=dict(l=0, r=0, t=20, b=0),
        font_size=20,
    )
    fig1.update_layout(**kw)
    fig2.update_layout(**kw)
    kw["yaxis"] = dict(range=[0.55, 1.15], title="(Test Bal Acc) / (Train Bal Acc)")
    fig3.update_layout(**kw)
    kw["yaxis"] = dict(range=[0.55, 1.0], title="Balanced Train Acc")
    fig4.update_layout(**kw)
    kw["yaxis"] = dict(range=[0.55, 1.0], title="Raw Test Acc")
    fig5.update_layout(**kw)

    # Adv vs Wyner
    # KS test
    print("#" * 120)
    print("COMPARE TEST BAL ACCS")
    print("*" * 80)
    print("Adv vs Density Ratio")
    x = best_bal_accs_each_method["adv"]
    y = best_bal_accs_each_method["wyner"]
    # Welch's t test
    statistic, pvalue = stats.ttest_ind(x, y, equal_var=False)
    print(f"Welch's t test: {statistic=:.3f}, {pvalue=:.5f}")
    symbol = choose_significance_symbol(pvalue)
    print()
    add_pvalue_annotation(fig1, 0, 1, 0.92, 0.94, symbol)

    # Adv vs Wasserstein
    # KS test
    print("*" * 80)
    print("Adv vs Wasserstein")
    x = best_bal_accs_each_method["adv"]
    y = best_bal_accs_each_method["wasserstein"]
    # Welch's t test
    statistic, pvalue = stats.ttest_ind(x, y, equal_var=False)
    print(f"Welch's t test, adv vs wasserstein, {statistic=:.3f}, {pvalue=:.5f}")
    symbol = choose_significance_symbol(pvalue)
    print()
    add_pvalue_annotation(fig1, 0, 2, 0.97, 0.99, symbol)

    # Subfigure title roughly: "Pooled performance of best $\lambda$ for each censor mode"
    fig1.write_image(
        figs_dir / "summary_boxplot_pool_best_lambda_each_mode.test_bal_acc.jpeg", width=500, height=500, scale=3
    )

    print("#" * 120)
    print("COMPARE VARIATION IN MEAN PERFORMANCE ACROSS ALL HYPERPARAMS")
    print("*" * 80)
    print("Adv vs Density Ratio")
    x = avgs_across_hparams_each_method["adv"]
    y = avgs_across_hparams_each_method["wyner"]
    # levene test for non-equal variance
    print(f"Variances: {np.var(x)=:.5f}, {np.var(y)=:.5f}")
    statistic, pvalue = stats.levene(x, y)
    print(f"Levene test for equal variance: {statistic=:.3f}, {pvalue=:.5f}")
    symbol = choose_significance_symbol(pvalue)
    print()
    add_pvalue_annotation(fig2, 0, 1, 0.92, 0.94, symbol)

    print("*" * 80)
    print("Adv vs Wasserstein")
    x = avgs_across_hparams_each_method["adv"]
    y = avgs_across_hparams_each_method["wasserstein"]
    print(f"Variances: {np.var(x)=:.5f}, {np.var(y)=:.5f}")
    statistic, pvalue = stats.levene(x, y)
    print(f"Levene test for equal variance: {statistic=:.3f}, {pvalue=:.5f}")
    symbol = choose_significance_symbol(pvalue)
    print()
    add_pvalue_annotation(fig2, 0, 2, 0.97, 0.99, symbol)

    # Subfig title, roughly: "Variability in performance across lambda values and censor modes"
    fig2.write_image(
        figs_dir / "summary_boxplot_means_from_each_hparam.test_bal_acc.jpeg", width=500, height=500, scale=3
    )

    print("#" * 120)
    print("COMPARE RATIOS OF TEST_BAL_ACC / TRAIN_BAL_ACC")
    # Adv vs Wyner
    # KS test
    print("COMPARE TEST BAL ACCS")
    print("*" * 80)
    print("Adv vs Density Ratio")
    x = best_test_train_ratios_each_method["adv"]
    y = best_test_train_ratios_each_method["wyner"]
    # Welch's t test
    statistic, pvalue = stats.ttest_ind(x, y, equal_var=False)
    print(f"Welch's t test: {statistic=:.3f}, {pvalue=:.5f}")
    symbol = choose_significance_symbol(pvalue)
    print()
    add_pvalue_annotation(fig3, 0, 1, 0.92, 0.94, symbol)

    # Adv vs Wasserstein
    # KS test
    print("*" * 80)
    print("Adv vs Wasserstein")
    x = best_test_train_ratios_each_method["adv"]
    y = best_test_train_ratios_each_method["wasserstein"]
    # Welch's t test
    statistic, pvalue = stats.ttest_ind(x, y, equal_var=False)
    print(f"Welch's t test: {statistic=:.3f}, {pvalue=:.5f}")
    symbol = choose_significance_symbol(pvalue)
    print()
    add_pvalue_annotation(fig3, 0, 2, 0.97, 0.99, symbol)

    # Subfig title, roughly: "Generalization ratios using best $\lambda$ from each censor mode"
    fig3.write_image(
        figs_dir / "summary_boxplot_pool_best_lambda_each_mode.test_train_ratios.jpeg", width=500, height=500, scale=3
    )

    print("#" * 120)
    print("COMPARE TRAIN BAL ACCS")
    # Adv vs Wyner
    # KS test
    print("*" * 80)
    print("Adv vs Density Ratio")
    x = train_accs_each_method["adv"]
    y = train_accs_each_method["wyner"]
    # Welch's t test
    statistic, pvalue = stats.ttest_ind(x, y, equal_var=False)
    print(f"Welch's t test: {statistic=:.3f}, {pvalue=:.5f}")
    symbol = choose_significance_symbol(pvalue)
    print()
    add_pvalue_annotation(fig3, 0, 1, 0.92, 0.94, symbol)

    # Adv vs Wasserstein
    # KS test
    print("*" * 80)
    print("Adv vs Wasserstein")
    x = train_accs_each_method["adv"]
    y = train_accs_each_method["wasserstein"]
    # Welch's t test
    statistic, pvalue = stats.ttest_ind(x, y, equal_var=False)
    print(f"Welch's t test: {statistic=:.3f}, {pvalue=:.5f}")
    symbol = choose_significance_symbol(pvalue)
    print()
    add_pvalue_annotation(fig3, 0, 2, 0.97, 0.99, symbol)

    # Subfig title, roughly: "Generalization ratios using best $\lambda$ from each censor mode"
    fig4.write_image(
        figs_dir / "summary_boxplot_pool_best_lambda_each_mode.train_accs.jpeg", width=500, height=500, scale=3
    )

    print("#" * 120)
    print("COMPARE RAW TEST ACCS")
    # Adv vs Wyner
    # KS test
    print("*" * 80)
    print("Adv vs Density Ratio")
    x = best_raw_accs_each_method["adv"]
    y = best_raw_accs_each_method["wyner"]
    # Welch's t test
    statistic, pvalue = stats.ttest_ind(x, y, equal_var=False)
    print(f"Welch's t test: {statistic=:.3f}, {pvalue=:.5f}")
    symbol = choose_significance_symbol(pvalue)
    print()
    add_pvalue_annotation(fig3, 0, 1, 0.92, 0.94, symbol)

    # Adv vs Wasserstein
    # KS test
    print("*" * 80)
    print("Adv vs Wasserstein")
    x = best_raw_accs_each_method["adv"]
    y = best_raw_accs_each_method["wasserstein"]
    # Welch's t test
    statistic, pvalue = stats.ttest_ind(x, y, equal_var=False)
    print(f"Welch's t test: {statistic=:.3f}, {pvalue=:.5f}")
    symbol = choose_significance_symbol(pvalue)
    print()
    add_pvalue_annotation(fig3, 0, 2, 0.97, 0.99, symbol)

    # Subfig title, roughly: "Generalization ratios using best $\lambda$ from each censor mode"
    fig5.write_image(
        figs_dir / "summary_boxplot_pool_best_lambda_each_mode.raw_test_acc.jpeg", width=500, height=500, scale=3
    )


def scatter_test_vs_testOverTrain(df, figs_dir):
    # Scatter plot of (test_acc vs test_acc/train_acc)

    B_df = df[df["name"] == "baseline"]

    B_test_over_train = B_df[B_df["split"] == "test_over_train"]
    B_test = B_df[B_df["split"] == "test"]
    B_merged = B_test_over_train.merge(
        B_test, left_on=["fold", "seed"], right_on=["fold", "seed"], suffixes=["_x", "_y"]
    )
    assert len(B_test_over_train) == 100 and len(B_test) == 100 and len(B_merged) == 100

    # Note that baseline has weight "None" which we want to exclude
    n_colors = df["weight"].unique().size - 1
    colors = px.colors.sample_colorscale(colorscale, n_colors)

    mean_kw = dict(
        marker_opacity=0.8,
        marker_size=15,
        marker_symbol="x",
        marker_line_width=2,
        marker_line_color="black",
    )
    dot_kw = dict(
        mode="markers",
        marker_opacity=0.7,
        marker_size=5,
    )

    box_kw = dict(
        line_width=2,
        line_color="black",
        opacity=0.4,
    )

    # gather figures into list, so all axes can be scaled together
    all_figs_titles = []

    x_lo = y_lo = np.inf
    x_hi = y_hi = -np.inf

    def update_xy(x, y, curr_x_lo, curr_x_hi, curr_y_lo, curr_y_hi):
        new_x_lo, new_x_hi = np.min(x), np.max(x)
        new_y_lo, new_y_hi = np.min(y), np.max(y)

        new_x_lo, new_x_hi = np.quantile(x, q=[0.20, 0.80])
        new_y_lo, new_y_hi = np.quantile(y, q=[0.20, 0.80])

        final_x_lo = min(curr_x_lo, new_x_lo)
        final_y_lo = min(curr_y_lo, new_y_lo)

        final_x_hi = max(curr_x_hi, new_x_hi)
        final_y_hi = max(curr_y_hi, new_y_hi)

        return final_x_lo, final_x_hi, final_y_lo, final_y_hi

    for method in ["adv", "wyner", "wasserstein"]:
        method_df = df[df["method"] == method]
        for name, name_group in method_df.groupby("name", sort=False):
            fig = go.Figure()

            # Add baseline
            x = B_merged["bal_acc_x"].values
            y = B_merged["bal_acc_y"].values
            fig.add_trace(go.Scatter(y=y, x=x, name="$\lambda=0$", marker_color="red", marker_symbol="cross", **dot_kw))

            # Add mean of baseline
            kw = dict(mode="markers", showlegend=False, marker_color="red")
            fig.add_trace(go.Scatter(y=[np.mean(y)], x=[np.mean(x)], **kw, **mean_kw))

            # Add box for baseline
            y_q25, y_q75 = np.quantile(y, q=[0.25, 0.75])
            x_q25, x_q75 = np.quantile(x, q=[0.25, 0.75])
            fig.add_shape(fillcolor="red", x0=x_q25, x1=x_q75, y0=y_q25, y1=y_q75, **box_kw)

            # Update axis range for baseline:
            x_lo, x_hi, y_lo, y_hi = update_xy(x, y, x_lo, x_hi, y_lo, y_hi)

            # # Add y=-x + C line passing through baseline
            # kw = dict(showlegend=False, mode="lines", line=dict(width=2, dash="dash", color="rgba(0, 0, 0, 0.5)"))
            # C = np.mean(x) + np.mean(y)
            # fig.add_trace(go.Scatter(x=[0, C], y=[C, 0], **kw))

            # Add scatter group for each value of lambda
            for i, (weight, weight_group) in enumerate(name_group.groupby("weight", sort=False)):
                weight_group_x = weight_group[weight_group["split"] == "test_over_train"]
                weight_group_y = weight_group[weight_group["split"] == "test"]
                weight_group_merged = weight_group_x.merge(
                    weight_group_y, left_on=["fold", "seed"], right_on=["fold", "seed"], suffixes=["_x", "_y"]
                )
                assert len(weight_group_x) == 100 and len(weight_group_y) == 100 and len(weight_group_merged) == 100
                x = weight_group_merged["bal_acc_x"]
                y = weight_group_merged["bal_acc_y"]

                # Add points
                fig.add_trace(go.Scatter(y=y, x=x, name=rf"$\lambda={weight}$", marker_color=colors[i], **dot_kw))

                # Update axis range for points:
                x_lo, x_hi, y_lo, y_hi = update_xy(x, y, x_lo, x_hi, y_lo, y_hi)

                # Add mean
                kw = dict(mode="markers", showlegend=False, marker_color=colors[i])
                fig.add_trace(go.Scatter(y=[np.mean(y)], x=[np.mean(x)], **kw, **mean_kw))

                # Add box
                y_q25, y_q75 = np.quantile(y, q=[0.25, 0.75])
                x_q25, x_q75 = np.quantile(x, q=[0.25, 0.75])
                fig.add_shape(x0=x_q25, x1=x_q75, y0=y_q25, y1=y_q75, fillcolor=colors[i], **box_kw)

            # # Add y=x line
            # kw = dict(showlegend=False, mode="lines", line=dict(width=2, dash="dash", color="rgba(0, 0, 0, 0.5)"))
            # fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], **kw))

            title = f"scatter_test_vs_trainOverTest__method={method}_name={name}.jpeg"
            fig.update_layout(
                # title=title,
                template=layout_template,
                yaxis_title="Test Bal Acc",
                xaxis=dict(
                    title="(Test Bal Acc) / (Train Bal Acc)",
                    constrain="domain",
                    # scaleanchor="y",
                    # scaleratio=1,
                ),
                legend=dict(orientation="h", yanchor="bottom", y=1.00, xanchor="right", x=1, itemsizing="constant"),
                width=800,
                height=800,
                margin=dict(l=0, r=0, t=10, b=0),
                font_size=20,
            )
            all_figs_titles.append((fig, title))

    print("Adaptive axis ranges would be:")
    print("x", x_lo, x_hi)
    print("y", y_lo, y_hi)

    # NOTE - using a fixed axis range, based on the subset of plots displayed in paper
    x_lo, x_hi = [0.65, 1.05]
    y_lo, y_hi = [0.5, 0.8]
    print("Using fixed axis ranges:")
    print("x", x_lo, x_hi)
    print("y", y_lo, y_hi)
    for fig, title in all_figs_titles:
        fig.update_layout(xaxis_range=[x_lo, x_hi], yaxis_range=[y_lo, y_hi])
        fig.write_image(figs_dir / title, width=800, height=800, scale=3)
        # fig.show()


def load(figs_dir, experiment_name):
    with open(figs_dir / f"preprocessed__{experiment_name}.pkl", "rb") as f:
        df, train_records, test_records, censored_keys = pickle.load(f)
    return df, train_records, test_records, censored_keys


def main(experiment_name, which_ckpt, results_dir, figs_dir):
    df, _train_records, _test_records, _censored_keys = load(figs_dir, experiment_name)

    print("Comparisons across methods")
    compare_methods_overall(df, figs_dir)

    print("Boxplot of test acc...")
    boxplot_test_acc(df, figs_dir)

    print("Scatter test vs test/train acc...")
    scatter_test_vs_testOverTrain(df, figs_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", required=True)
    args = parser.parse_args()
    seed_everything(0)

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
