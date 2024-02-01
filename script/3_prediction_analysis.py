#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %%
# Imports

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

# from utils.yamlhandler import YamlHandler  # Configuration
from utils.args import getparam

import utils

from sklearn import metrics as skmetrics
from utils.metrics import total_sigma, within_sigma, between_sigma

from scipy import stats

import os
import sys

from itertools import product as iterprod

plt.style.use('seaborn')

pd.set_option('display.precision', 3)
pd.set_option('display.width', 200)
sns.set_context('paper', font_scale=2)

savefig_kwargs = dict(dpi=300)

# %%
# Main Paths

cur_dir = os.getcwd()
parent_dir = os.path.dirname(cur_dir)

# %%
# Config name

config_name = getparam('config', 'deeplearn')
which_region = getparam('region', 'geysers')

# %%
# Plot options

show_fig = True  # False
save_plot = False  # True

# %%
# Derived paths from the config name

results_path = os.path.join(parent_dir, 'results', config_name)

results_comparison_path = os.path.join(results_path, 'comparison')

results_plots_path = os.path.join(results_path, 'plots')
os.makedirs(results_plots_path, exist_ok=True)

# %%
# Features, targets and event

features_cols = ['Mw', 'Rhypo']
target_names = ['PGV', 'PGA', 'SA-0.2', 'SA-0.5', 'SA-1.0']
target_labels = {
    'PGV': 'PGV (m/s)', 'PGA': 'PGA (m/s²)',
    'SA-0.2': 'SA (T=0.2s) (m/s²)',
    'SA-0.5': 'SA (T=0.5s) (m/s²)',
    'SA-1.0': 'SA (T=1.0s) (m/s²)'}

s_targets = {target: f's({target})' for target in target_names}

# %%
# Load the data

df_trainval = pd.read_pickle(
    os.path.join(parent_dir, 'dataset', 'features', f'Xy_{which_region}_{config_name}_train.pkl.bz2'))
df_test = pd.read_pickle(
    os.path.join(parent_dir, 'dataset', 'features', f'Xy_{which_region}_{config_name}_test.pkl.bz2'))

df_trainval['event'] = df_trainval.index.get_level_values('event')
df_test['event'] = df_test.index.get_level_values('event')
dfs = {
    'overall': pd.concat([df_trainval, df_test]),
    'trainval': df_trainval, 'test': df_test}

# %%
# The results we are interested in.

which_methods = ['MOD3', 'deeplearn']
which_methods_names = {
    'MOD1': 'MOD1',
    'MOD3': 'MOD3',
    'deeplearn': 'ANN',
}

namedfs = ['trainval', 'test']

# %%
# Generate the limit values for equally distributed plots

lims_truepred = {target: [0, 0] for target in target_names}
lims_resid = {target: [0, 0] for target in target_names}


target = target_names[1]
namedf = namedfs[0]
method = which_methods[0]
for target, namedf, method in iterprod(target_names, namedfs, which_methods):
    log_target = f'log10({target})'
    pred_log_target = f'{log_target}_pred'
    resid_log_target = f'{log_target}_resid'

    dff_features = dfs[namedf][features_cols + ['event']]

    if namedf == 'overall':
        y_true = pd.concat([pd.read_pickle(os.path.join(
            results_comparison_path, f'y_true_{n}_{target}.pkl'))
            for n in ['trainval', 'test']])
        ys_pred = pd.concat([pd.read_pickle(os.path.join(
            results_comparison_path, f'y_pred_{n}_{target}_{method}.pkl'))
            for n in ['trainval', 'test']])
    else:
        y_true: pd.Series = pd.read_pickle(os.path.join(
            results_comparison_path, f'y_true_{namedf}_{target}.pkl'))
        ys_pred = pd.read_pickle(os.path.join(
            results_comparison_path, f'y_pred_{namedf}_{target}_{method}.pkl'))

    y_pred_mean = ys_pred.mean(axis=1).rename(pred_log_target)

    dff = pd.concat([dff_features, y_true, ys_pred, y_pred_mean], axis=1)

    dff[resid_log_target] = dff[log_target] - dff[pred_log_target]

    lims_truepred_sub = [
        utils.round_to(
            dff[[log_target, pred_log_target]].values.min() - 0.5, 0.5),
        utils.round_to(
            dff[[log_target, pred_log_target]].values.max() + 0.5, 0.5)]

    lims_truepred_target = lims_truepred[target]

    lims_truepred_target = [
        np.min([lims_truepred_sub[0], lims_truepred_target[0]]),
        np.max([lims_truepred_sub[1], lims_truepred_target[1]])]

    lims_truepred[target] = lims_truepred_target

    lims_resid_sub = [
        utils.round_to(dff[resid_log_target].min(), 0.5),
        utils.round_to(dff[resid_log_target].max(), 0.5)]

    max_abs = np.max(np.abs(lims_resid_sub))

    lims_resid_target = lims_resid[target]

    lims_resid_target = [
        np.min([-max_abs, lims_resid_target[0]]),
        np.max([max_abs, lims_resid_target[1]])]

    lims_resid[target] = lims_resid_target

# %%
# Run the plots

n_bins = 10

target = target_names[0]
namedf = namedfs[0]
method = which_methods[0]
for target, namedf, method in iterprod(target_names, namedfs, which_methods):

    method2 = which_methods_names[method]
    print(f'Considering target: {target}, name: {namedf}, method: {method}')

    log_target = f'log10({target})'
    pred_log_target = f'{log_target}_pred'
    resid_log_target = f'{log_target}_resid'
    pred_target = f'{target}_pred'
    s_target = f's({target})'

    plots_path = os.path.join(results_plots_path, namedf, target)
    os.makedirs(plots_path, exist_ok=True)

    dff_features = dfs[namedf][
        features_cols + target_names + ['event'] + ['DEPT'] + [s_target]]

    if namedf == 'overall':
        y_true = pd.concat([pd.read_pickle(os.path.join(
            results_comparison_path, f'y_true_{n}_{target}.pkl'))
            for n in ['trainval', 'test']])
        ys_pred = pd.concat([pd.read_pickle(os.path.join(
            results_comparison_path, f'y_pred_{n}_{target}_{method}.pkl'))
            for n in ['trainval', 'test']])
    else:
        y_true: pd.Series = pd.read_pickle(os.path.join(
            results_comparison_path, f'y_true_{namedf}_{target}.pkl'))
        ys_pred = pd.read_pickle(os.path.join(
            results_comparison_path, f'y_pred_{namedf}_{target}_{method}.pkl'))

    y_pred_mean = ys_pred.mean(axis=1).rename(pred_log_target)

    dff = pd.concat([dff_features, y_true, ys_pred, y_pred_mean], axis=1)

    dff[resid_log_target] = dff[log_target] - dff[pred_log_target]
    dff[pred_target] = np.power(10, dff[pred_log_target])

    r2_score = skmetrics.r2_score(dff[log_target], dff[pred_log_target])

    # Plot distribution True-Pred with density
    values_kde = dff[[log_target, pred_log_target]].values

    kernel = stats.gaussian_kde(values_kde.T)(values_kde.T)

    kernel_range = [kernel.min(), kernel.max()]
    norm_kernel = plt.Normalize(*kernel_range)
    sm_kernel = plt.cm.ScalarMappable(cmap='viridis', norm=norm_kernel)
    sm_kernel.set_array([])

    lims_truepred_target = lims_truepred[target]

    fig, ax = plt.subplots(1, 1)
    ax.axline((-3, -3), slope=1, linestyle='--', c='black')
    sns.scatterplot(
        data=dff, x=pred_log_target, y=log_target, ax=ax, alpha=0.33,
        c=kernel, cmap=sm_kernel.cmap, edgecolor='none')
    # sns.kdeplot(
    #     data=dff, x=pred_log_target, y=log_target, ax=ax, levels=5,
    #     fill=True, alpha=0.4, cut=1, cmap='viridis')
    fig.suptitle(f'Method: {method2}')
    ax.text(0.05, 0.95, f'R^2: {r2_score:.3f}', transform=ax.transAxes)
    cbar = fig.colorbar(sm_kernel, ticks=kernel_range, label='Density')
    cbar.ax.set_yticklabels(['Low', 'High'])
    ax.set_xlim(lims_truepred_target)
    ax.set_ylim(lims_truepred_target)
    if save_plot:
        fig.savefig(os.path.join(
            plots_path, f'true_vs_pred-density-{method2}.png'),
            **savefig_kwargs)
    if show_fig:
        plt.show()
    plt.close()

    # Plot distribution True-Pred with Mw colored
    norm_Mw = plt.Normalize(dff['Mw'].min(), dff['Mw'].max())
    sm_Mw = plt.cm.ScalarMappable(cmap='magma_r', norm=norm_Mw)
    sm_Mw.set_array([])

    fig, ax = plt.subplots(1, 1)
    ax.axline((-3, -3), slope=1, linestyle='--', c='black')
    sns.scatterplot(
        data=dff, x=pred_log_target, y=log_target, ax=ax, alpha=0.33,
        hue='Mw', palette=sm_Mw.cmap, edgecolor='none')
    fig.suptitle(f'Method: {method2}')
    ax.set_xlim(lims_truepred_target)
    ax.set_ylim(lims_truepred_target)
    # ax.text(0.05, 0.95, f'R^2: {r2_score:.3f}', transform=ax.transAxes)
    ax.get_legend().remove()
    fig.colorbar(sm_Mw, label='Mw')
    if save_plot:
        fig.savefig(os.path.join(
            plots_path, f'true_vs_pred-hueMw-{method2}.png'),
            **savefig_kwargs)
    if show_fig:
        plt.show()
    plt.close()

    # Plot distribution True-Pred with Rhypo colored
    norm_Rhypo = plt.Normalize(dff['Rhypo'].min(), dff['Rhypo'].max())
    sm_Rhypo = plt.cm.ScalarMappable(cmap='crest', norm=norm_Rhypo)
    sm_Rhypo.set_array([])

    fig, ax = plt.subplots(1, 1)
    ax.axline((-3, -3), slope=1, linestyle='--', c='black')
    sns.scatterplot(
        data=dff, x=pred_log_target, y=log_target, ax=ax, alpha=0.33,
        hue='Rhypo', palette=sm_Rhypo.cmap, edgecolor='none')
    fig.suptitle(f'Method: {method2}')
    ax.set_xlim(lims_truepred_target)
    ax.set_ylim(lims_truepred_target)
    ax.get_legend().remove()
    fig.colorbar(sm_Rhypo, label='Rhypo')
    if save_plot:
        fig.savefig(os.path.join(
            plots_path, f'true_vs_pred-hueRhypo-{method2}.png'),
            **savefig_kwargs)
    if show_fig:
        plt.show()
    plt.close()

    df_mean_resid = \
        dff[[resid_log_target]].groupby(level='event').mean().dropna()

    lims_resid_target = lims_resid[target]

    # Grouped means and stds for Mw
    means_Mw = dff[[resid_log_target, 'Mw']].groupby('Mw').mean()
    stds_Mw = dff[[resid_log_target, 'Mw']].groupby('Mw').std()
    min_stds_Mw = means_Mw - stds_Mw
    max_stds_Mw = means_Mw + stds_Mw
    min_lines_Mw = means_Mw.index.values - 0.02
    max_lines_Mw = means_Mw.index.values + 0.02

    if namedf == 'trainval':
        xticks_Mw = np.arange(1.25, 3.5, 0.25)
    else:
        xticks_Mw = np.arange(0.75, 3.5, 0.25)

    # Plot distribution Residual with Mw, with grouped means and stds

    fig, ax = plt.subplots(1, 1)
    sns.scatterplot(
        data=dff, x='Mw', y=resid_log_target, ax=ax, alpha=0.4,
        edgecolor='none')
    ax.axhline(0, linestyle='--', c='black')
    sns.scatterplot(
        x=means_Mw.index.values, y=means_Mw.iloc[:, 0].values, color='black',
        edgecolor='none', s=30)
    ax.vlines(
        means_Mw.index.values - 0.001, ymin=min_stds_Mw, ymax=max_stds_Mw,
        colors='black', linewidths=2.5)
    for i in range(len(min_stds_Mw)):
        ax.hlines(
            [min_stds_Mw.iloc[i].values[0], max_stds_Mw.iloc[i].values[0]],
            xmin=min_lines_Mw[i], xmax=max_lines_Mw[i], colors='black')
    ax.set_xticks(xticks_Mw)
    fig.suptitle(f'Method: {method2}')
    ax.set_ylim(lims_resid_target)
    if save_plot:
        fig.savefig(
            os.path.join(plots_path, f'resid-Mw-grouped-{method2}.png'),
            **savefig_kwargs)
    if show_fig:
        plt.show()
    plt.close()

    # Plot distribution Residual with Mw, colored by Rhypo
    fig, ax = plt.subplots(1, 1)
    sns.scatterplot(
        data=dff, x='Mw', y=resid_log_target, ax=ax, alpha=0.4,
        edgecolor='none', hue='Rhypo', palette=sm_Rhypo.cmap)
    ax.axhline(0, linestyle='--', c='black')
    ax.set_xticks(xticks_Mw)
    fig.suptitle(f'Method: {method2}')
    ax.set_ylim(lims_resid_target)
    ax.get_legend().remove()
    fig.colorbar(sm_Rhypo, label='Rhypo')
    if save_plot:
        fig.savefig(os.path.join(plots_path, f'resid-Mw-{method2}.png'),
                    **savefig_kwargs)
    if show_fig:
        plt.show()
    plt.close()

    # Grouped means and stds for Rhypo
    bins_Rhypo = 10 ** np.linspace(
        np.log10(dff_features['Rhypo'].min() - 0.0001),
        np.log10(dff_features['Rhypo'].max() + 0.0001), n_bins + 1)
    bins_Rhypo = np.linspace(
        dff_features['Rhypo'].min() - 0.0001,
        dff_features['Rhypo'].max() + 0.0001, n_bins + 1)

    bin_means, bin_edges_means, binnumber_means = stats.binned_statistic(
        x=dff['Rhypo'], values=dff[resid_log_target], bins=bins_Rhypo,
        statistic='mean')
    bin_stds, bin_edges_stds, binnumber_stds = stats.binned_statistic(
        x=dff['Rhypo'], values=dff[resid_log_target], bins=bins_Rhypo,
        statistic='std')

    assert np.all(bin_edges_means == bin_edges_stds)
    assert np.all(binnumber_means == binnumber_stds)

    bin_width = (bin_edges_means[1] - bin_edges_means[0])
    bin_centers = bin_edges_means[1:] - bin_width / 2

    min_stds_Rhypo = bin_means - bin_stds
    max_stds_Rhypo = bin_means + bin_stds
    min_lines_Rhypo = bin_means - 0.02
    max_lines_Rhypo = bin_means + 0.02

    # Plot distribution Residual with Rhypo, with grouped means and stds
    fig, ax = plt.subplots(1, 1)
    sns.scatterplot(
        data=dff, x='Rhypo', y=resid_log_target, ax=ax, alpha=0.4,
        edgecolor='none')
    ax.axhline(0, linestyle='--', c='black')
    sns.scatterplot(
        x=bin_centers, y=bin_means, color='black',
        edgecolor='none', s=30)
    ax.vlines(
        bin_centers, ymin=min_stds_Rhypo, ymax=max_stds_Rhypo,
        colors='black', linewidths=2.5)
    for i in range(len(min_stds_Rhypo)):
        ax.hlines(
            [min_stds_Rhypo[i], max_stds_Rhypo[i]],
            xmin=min_lines_Rhypo[i], xmax=max_lines_Rhypo[i], colors='black')
    fig.suptitle(f'Method: {method2}')
    ax.set_ylim(lims_resid_target)
    ax.set_xticks(np.arange(0, 21, 2))
    if save_plot:
        fig.savefig(os.path.join(
            plots_path, f'resid-Rhypo-grouped-{method2}.png'),
            **savefig_kwargs)
    if show_fig:
        plt.show()
    plt.close()

    # Grouped means and stds for depth
    bins_DEPT = 10 ** np.linspace(
        np.log10(dff_features['DEPT'].min() - 0.0001),
        np.log10(dff_features['DEPT'].max() + 0.0001), n_bins + 1)
    bins_DEPT = np.linspace(
        dff_features['DEPT'].min() - 0.0001,
        dff_features['DEPT'].max() + 0.0001, n_bins + 1)

    bin_means, bin_edges_means, binnumber_means = stats.binned_statistic(
        x=dff['DEPT'], values=dff[resid_log_target], bins=bins_DEPT,
        statistic='mean')
    bin_stds, bin_edges_stds, binnumber_stds = stats.binned_statistic(
        x=dff['DEPT'], values=dff[resid_log_target], bins=bins_DEPT,
        statistic='std')

    assert np.all(bin_edges_means == bin_edges_stds)
    assert np.all(binnumber_means == binnumber_stds)

    bin_width = (bin_edges_means[1] - bin_edges_means[0])
    bin_centers = bin_edges_means[1:] - bin_width / 2

    min_stds_DEPT = bin_means - bin_stds
    max_stds_DEPT = bin_means + bin_stds
    min_lines_DEPT = bin_centers - 0.02
    max_lines_DEPT = bin_centers + 0.02

    # Plot distribution Residual with Depth, with grouped means and stds
    fig, ax = plt.subplots(1, 1)
    sns.scatterplot(
        data=dff, x='DEPT', y=resid_log_target, ax=ax, alpha=0.4,
        edgecolor='none')
    ax.axhline(0, linestyle='--', c='black')
    sns.scatterplot(
        x=bin_centers, y=bin_means, color='black',
        edgecolor='none', s=30)
    ax.vlines(
        bin_centers, ymin=min_stds_DEPT, ymax=max_stds_DEPT,
        colors='black', linewidths=2.5)
    for i in range(len(min_stds_DEPT)):
        ax.hlines(
            [min_stds_DEPT[i], max_stds_DEPT[i]],
            xmin=min_lines_DEPT[i], xmax=max_lines_DEPT[i], colors='black')
    fig.suptitle(f'Method: {method2}')
    ax.set_ylim(lims_resid_target)
    ax.set_xticks(np.arange(0, 3.1, 0.3))
    if save_plot:
        fig.savefig(os.path.join(
            plots_path, f'resid-DEPT-grouped-{method2}.png'),
            **savefig_kwargs)
    if show_fig:
        plt.show()
    plt.close()

# %%
# Plot binned residuals unified

n_bins = 10

target = target_names[0]
namedf = namedfs[0]
for namedf in namedfs:
    if namedf == 'trainval':
        xticks_Mw = np.arange(1.25, 3.5, 0.25)
    else:
        xticks_Mw = np.arange(0.75, 3.5, 0.25)
    xticks_Rhypo = np.arange(0, 21, 2)

    method = which_methods[0]
    for method in which_methods:
        method2 = which_methods_names[method]
        print(f'Considering name: {namedf}, method: {method}')

        fig, axs = plt.subplots(
            len(target_names), 2, figsize=(15, 18), sharey='row', sharex='col')
        dff_features = dfs[namedf][
            features_cols + target_names + ['event']
            + list(s_targets.values())]

        plots_path = os.path.join(results_plots_path, namedf, 'alltargets')
        os.makedirs(plots_path, exist_ok=True)
        target = target_names[0]
        lims_resid_all = max(lims_resid.values(), key=lambda v: v[1])
        for itarget, target in enumerate(target_names):
            log_target = f'log10({target})'
            pred_log_target = f'{log_target}_pred'
            resid_log_target = f'{log_target}_resid'
            pred_target = f'{target}_pred'
            s_target = s_targets[target]
            if namedf == 'overall':
                y_true = pd.concat([pd.read_pickle(os.path.join(
                    results_comparison_path, f'y_true_{n}_{target}.pkl'))
                    for n in ['trainval', 'test']])
                ys_pred = pd.concat([pd.read_pickle(os.path.join(
                    results_comparison_path,
                    f'y_pred_{n}_{target}_{method}.pkl'))
                    for n in ['trainval', 'test']])
            else:
                y_true: pd.Series = pd.read_pickle(os.path.join(
                    results_comparison_path, f'y_true_{namedf}_{target}.pkl'))
                ys_pred = pd.read_pickle(os.path.join(
                    results_comparison_path,
                    f'y_pred_{namedf}_{target}_{method}.pkl'))

            y_pred_mean = ys_pred.mean(axis=1).rename(pred_log_target)

            dff = pd.concat(
                [dff_features, y_true, ys_pred, y_pred_mean], axis=1)

            dff[resid_log_target] = dff[log_target] - dff[pred_log_target]
            dff[pred_target] = np.power(10, dff[pred_log_target])

            # Grouped means and stds for Mw
            means_Mw = dff[[resid_log_target, 'Mw']].groupby('Mw').mean()
            stds_Mw = dff[[resid_log_target, 'Mw']].groupby('Mw').std()
            min_stds_Mw = means_Mw - stds_Mw
            max_stds_Mw = means_Mw + stds_Mw
            min_lines_Mw = means_Mw.index.values - 0.02
            max_lines_Mw = means_Mw.index.values + 0.02

            # Grouped means and stds for Rhypo
            bins_Rhypo = 10 ** np.linspace(
                np.log10(dff_features['Rhypo'].min() - 0.0001),
                np.log10(dff_features['Rhypo'].max() + 0.0001), n_bins + 1)
            bins_Rhypo = np.linspace(
                dff_features['Rhypo'].min() - 0.0001,
                dff_features['Rhypo'].max() + 0.0001, n_bins + 1)

            bin_means, bin_edges_means, binnumber_means = \
                stats.binned_statistic(
                    x=dff['Rhypo'], values=dff[resid_log_target],
                    bins=bins_Rhypo, statistic='mean')
            bin_stds, bin_edges_stds, binnumber_stds = \
                stats.binned_statistic(
                    x=dff['Rhypo'], values=dff[resid_log_target],
                    bins=bins_Rhypo, statistic='std')

            assert np.all(bin_edges_means == bin_edges_stds)
            assert np.all(binnumber_means == binnumber_stds)

            bin_width = (bin_edges_means[1] - bin_edges_means[0])
            bin_centers = bin_edges_means[1:] - bin_width / 2

            min_stds_Rhypo = bin_means - bin_stds
            max_stds_Rhypo = bin_means + bin_stds
            min_lines_Rhypo = bin_means - 0.02
            max_lines_Rhypo = bin_means + 0.02

            ax = axs[itarget, 0]
            # Plot distribution Residual with Rhypo, with grouped means and stds
            sns.scatterplot(
                data=dff, x='Rhypo', y=resid_log_target, ax=ax, alpha=0.4,
                edgecolor='none')
            ax.axhline(0, linestyle='--', c='black')
            sns.scatterplot(
                x=bin_centers, y=bin_means, color='black',
                edgecolor='none', s=60, ax=ax)
            ax.vlines(
                bin_centers, ymin=min_stds_Rhypo, ymax=max_stds_Rhypo,
                colors='black', linewidths=2.5)
            for i in range(len(min_stds_Rhypo)):
                ax.hlines(
                    [min_stds_Rhypo[i], max_stds_Rhypo[i]],
                    xmin=min_lines_Rhypo[i], xmax=max_lines_Rhypo[i],
                    colors='black')
            ax.text(0.99, 0.9, target_labels[target], horizontalalignment='right',
                    transform=ax.transAxes)
            ax.set_ylim(lims_resid_all)
            ax.set_xticks(xticks_Rhypo)
            ax.set(xlabel='Hypocentral distance (km)', ylabel='Residuals')

            ax = axs[itarget, 1]
            # Plot distribution Residual with Mw, with grouped means and stds
            sns.scatterplot(
                data=dff, x='Mw', y=resid_log_target, ax=ax, alpha=0.4,
                edgecolor='none', s=80)
            ax.axhline(0, linestyle='--', c='black')
            sns.scatterplot(
                x=means_Mw.index.values, y=means_Mw.iloc[:, 0].values,
                color='black', edgecolor='none', s=60, ax=ax)
            ax.vlines(
                means_Mw.index.values - 0.001, ymin=min_stds_Mw, ymax=max_stds_Mw,
                colors='black', linewidths=2.5)
            for i in range(len(min_stds_Mw)):
                ax.hlines(
                    [min_stds_Mw.iloc[i].values[0], max_stds_Mw.iloc[i].values[0]],
                    xmin=min_lines_Mw[i], xmax=max_lines_Mw[i], colors='black')
            ax.text(0.01, 0.9, target_labels[target], transform=ax.transAxes)
            ax.set_xticks(xticks_Mw)
            ax.set_ylim(lims_resid_all)
            ax.set(xlabel=r'Magnitude ($M_w$)', ylabel='Residuals')

        # fig.suptitle(f'Method: {method2}')
        fig.tight_layout()
        if save_plot:
            fig.savefig(os.path.join(
                plots_path, f'resid-grouped-{method2}.png'), **savefig_kwargs)
        if show_fig:
            plt.show()
        plt.close()

# %%
# NEW Plot binned residuals unified (swapped indexes)

y_range = [-2, 2]
n_bins = 10
avg_aggreg = 'mean'

red_color = sns.color_palette('tab10', 10)[3]

y_range_s = '_yrange=' + '_'.join([str(s) for s in y_range]) if y_range is not None else ''

target = target_names[0]
namedf = namedfs[0]
for namedf in namedfs:

    plots_path = os.path.join(results_plots_path, namedf, 'alltargets_v2')
    os.makedirs(plots_path, exist_ok=True)

    if namedf == 'test':
        xticks_Rhypo = np.arange(0, 17, 2)
    else:
        xticks_Rhypo = np.arange(0, 21, 2)

    xticks_Mw_range = xticks_Mw[-1] - xticks_Mw[0]
    xticks_Rhypo_range = xticks_Rhypo[-1] - xticks_Rhypo[0]

    fig, axs = plt.subplots(
        len(target_names), len(which_methods), figsize=(15, 18),
        sharey='row', sharex='col')

    imethod, method = list(enumerate(which_methods))[0]
    for imethod, method in enumerate(which_methods):
        method2 = which_methods_names[method]
        print(f'Considering name: {namedf}, method: {method}')
        dff_features = dfs[namedf][
            features_cols + target_names + ['event']
            + list(s_targets.values())]
        target = target_names[0]

        if y_range is None:
            lims_resid_all = max(lims_resid.values(), key=lambda v: v[1])
        else:
            lims_resid_all = y_range

        for itarget, target in enumerate(target_names):
            log_target = f'log10({target})'
            pred_log_target = f'{log_target}_pred'
            resid_log_target = f'{log_target}_resid'
            pred_target = f'{target}_pred'
            s_target = s_targets[target]
            if namedf == 'overall':
                y_true = pd.concat([pd.read_pickle(os.path.join(
                    results_comparison_path, f'y_true_{n}_{target}.pkl'))
                    for n in ['trainval', 'test']])
                ys_pred = pd.concat([pd.read_pickle(os.path.join(
                    results_comparison_path,
                    f'y_pred_{n}_{target}_{method}.pkl'))
                    for n in ['trainval', 'test']])
            else:
                y_true: pd.Series = pd.read_pickle(os.path.join(
                    results_comparison_path, f'y_true_{namedf}_{target}.pkl'))
                ys_pred = pd.read_pickle(os.path.join(
                    results_comparison_path,
                    f'y_pred_{namedf}_{target}_{method}.pkl'))

            y_pred_mean = ys_pred.mean(axis=1).rename(pred_log_target)

            dff = pd.concat(
                [dff_features, y_true, ys_pred, y_pred_mean], axis=1)

            dff[resid_log_target] = dff[log_target] - dff[pred_log_target]
            dff[pred_target] = np.power(10, dff[pred_log_target])

            # Grouped means and stds for Rhypo
            bins_Rhypo = 10 ** np.linspace(
                np.log10(dff_features['Rhypo'].min() - 0.0001),
                np.log10(dff_features['Rhypo'].max() + 0.0001), n_bins + 1)
            bins_Rhypo = np.linspace(
                dff_features['Rhypo'].min() - 0.0001,
                dff_features['Rhypo'].max() + 0.0001, n_bins + 1)

            bin_means, bin_edges_means, binnumber_means = \
                stats.binned_statistic(
                    x=dff['Rhypo'], values=dff[resid_log_target],
                    bins=bins_Rhypo, statistic=avg_aggreg)
            bin_stds, bin_edges_stds, binnumber_stds = \
                stats.binned_statistic(
                    x=dff['Rhypo'], values=dff[resid_log_target],
                    bins=bins_Rhypo, statistic='std')

            assert np.all(bin_edges_means == bin_edges_stds)
            assert np.all(binnumber_means == binnumber_stds)

            bin_width = (bin_edges_means[1] - bin_edges_means[0])
            bin_centers = bin_edges_means[1:] - bin_width / 2

            min_stds_Rhypo = bin_means - bin_stds
            max_stds_Rhypo = bin_means + bin_stds
            min_lines_Rhypo = bin_centers - 0.01 * xticks_Rhypo_range
            max_lines_Rhypo = bin_centers + 0.01 * xticks_Rhypo_range

            ax = axs[itarget, imethod]
            # Plot distribution Residual with Rhypo, with grouped means
            # and stds
            sns.scatterplot(
                data=dff, x='Rhypo', y=resid_log_target, ax=ax, alpha=0.4,
                edgecolor='none', zorder=0)
            ax.vlines(
                bin_centers, ymin=min_stds_Rhypo, ymax=max_stds_Rhypo,
                colors='black', linewidths=2.5, zorder=1)
            for i in range(len(min_stds_Rhypo)):
                ax.hlines(
                    [min_stds_Rhypo[i], max_stds_Rhypo[i]],
                    xmin=min_lines_Rhypo[i], xmax=max_lines_Rhypo[i],
                    colors='black', zorder=1)
            ax.axhline(0, linestyle='--', c='black', zorder=1)
            sns.scatterplot(
                x=bin_centers, y=bin_means, color=red_color,
                edgecolor='none', s=60, ax=ax, zorder=2)
            ax.text(0.99, 0.9, target_labels[target],
                    horizontalalignment='right', transform=ax.transAxes)
            if itarget == 0:
                ax.set_title(f'{method2}')
            ax.set_ylim(lims_resid_all)
            ax.set_xticks(xticks_Rhypo)
            ax.set(xlabel='Hypocentral distance (km)', ylabel='Residuals')

    fig.tight_layout()
    if save_plot:
        fig.savefig(os.path.join(
            plots_path, f'resid-grouped_Rhypo{y_range_s}_bin={n_bins}_avg_aggreg={avg_aggreg}.png'), **savefig_kwargs)
    if show_fig:
        plt.show()
    plt.close()

target = target_names[0]
namedf = namedfs[0]
for namedf in namedfs:

    plots_path = os.path.join(results_plots_path, namedf, 'alltargets_v2')
    os.makedirs(plots_path, exist_ok=True)

    if namedf == 'trainval':
        xticks_DEPT = np.arange(0, 3.1, 0.5)
    else:
        xticks_DEPT = np.arange(0, 3.6, 0.5)
    xticks_DEPT_range = xticks_DEPT[-1] - xticks_DEPT[0]

    fig, axs = plt.subplots(
        len(target_names), len(which_methods), figsize=(15, 18),
        sharey='row', sharex='col')

    imethod, method = list(enumerate(which_methods))[0]
    for imethod, method in enumerate(which_methods):
        method2 = which_methods_names[method]
        print(f'Considering name: {namedf}, method: {method}')
        dff_features = dfs[namedf][
            ['DEPT'] + target_names + ['event']
            + list(s_targets.values())]
        target = target_names[0]

        if y_range is None:
            lims_resid_all = max(lims_resid.values(), key=lambda v: v[1])
        else:
            lims_resid_all = y_range

        for itarget, target in enumerate(target_names):
            log_target = f'log10({target})'
            pred_log_target = f'{log_target}_pred'
            resid_log_target = f'{log_target}_resid'
            pred_target = f'{target}_pred'
            s_target = s_targets[target]
            if namedf == 'overall':
                y_true = pd.concat([pd.read_pickle(os.path.join(
                    results_comparison_path, f'y_true_{n}_{target}.pkl'))
                    for n in ['trainval', 'test']])
                ys_pred = pd.concat([pd.read_pickle(os.path.join(
                    results_comparison_path,
                    f'y_pred_{n}_{target}_{method}.pkl'))
                    for n in ['trainval', 'test']])
            else:
                y_true: pd.Series = pd.read_pickle(os.path.join(
                    results_comparison_path, f'y_true_{namedf}_{target}.pkl'))
                ys_pred = pd.read_pickle(os.path.join(
                    results_comparison_path,
                    f'y_pred_{namedf}_{target}_{method}.pkl'))

            y_pred_mean = ys_pred.mean(axis=1).rename(pred_log_target)

            dff = pd.concat(
                [dff_features, y_true, ys_pred, y_pred_mean], axis=1)

            dff[resid_log_target] = dff[log_target] - dff[pred_log_target]
            dff[pred_target] = np.power(10, dff[pred_log_target])

            # Grouped means and stds for DEPT
            bins_DEPT = 10 ** np.linspace(
                np.log10(dff_features['DEPT'].min() - 0.0001),
                np.log10(dff_features['DEPT'].max() + 0.0001), n_bins + 1)
            bins_DEPT = np.linspace(
                dff_features['DEPT'].min() - 0.0001,
                dff_features['DEPT'].max() + 0.0001, n_bins + 1)

            bin_means, bin_edges_means, binnumber_means = \
                stats.binned_statistic(
                    x=dff['DEPT'], values=dff[resid_log_target],
                    bins=bins_DEPT, statistic=avg_aggreg)
            bin_stds, bin_edges_stds, binnumber_stds = \
                stats.binned_statistic(
                    x=dff['DEPT'], values=dff[resid_log_target],
                    bins=bins_DEPT, statistic='std')

            assert np.all(bin_edges_means == bin_edges_stds)
            assert np.all(binnumber_means == binnumber_stds)

            bin_width = (bin_edges_means[1] - bin_edges_means[0])
            bin_centers = bin_edges_means[1:] - bin_width / 2

            min_stds_DEPT = bin_means - bin_stds
            max_stds_DEPT = bin_means + bin_stds
            min_lines_DEPT = bin_centers - 0.01 * xticks_DEPT_range
            max_lines_DEPT = bin_centers + 0.01 * xticks_DEPT_range

            ax = axs[itarget, imethod]
            # Plot distribution Residual with DEPT, with grouped means
            # and stds
            sns.scatterplot(
                data=dff, x='DEPT', y=resid_log_target, ax=ax, alpha=0.4,
                edgecolor='none', zorder=0)
            ax.vlines(
                bin_centers, ymin=min_stds_DEPT, ymax=max_stds_DEPT,
                colors='black', linewidths=2.5, zorder=1)
            for i in range(len(min_stds_DEPT)):
                ax.hlines(
                    [min_stds_DEPT[i], max_stds_DEPT[i]],
                    xmin=min_lines_DEPT[i], xmax=max_lines_DEPT[i],
                    colors='black', zorder=1)
            ax.axhline(0, linestyle='--', c='black', zorder=1)
            sns.scatterplot(
                x=bin_centers, y=bin_means, color=red_color,
                edgecolor='none', s=60, ax=ax, zorder=2)
            ax.text(0.99, 0.9, target_labels[target],
                    horizontalalignment='right', transform=ax.transAxes)
            if itarget == 0:
                ax.set_title(f'{method2}')
            ax.set_ylim(lims_resid_all)
            ax.set_xticks(xticks_DEPT)
            ax.set(xlabel='Depth (km)', ylabel='Residuals')

    fig.tight_layout()
    if save_plot:
        fig.savefig(os.path.join(
            plots_path, f'resid-grouped_DEPT{y_range_s}_bin={n_bins}_avg_aggreg={avg_aggreg}.png'), **savefig_kwargs)
    if show_fig:
        plt.show()
    plt.close()

target = target_names[0]
namedf = namedfs[0]
for namedf in namedfs:
    if namedf == 'trainval':
        xticks_Mw = np.arange(1.25, 3.5, 0.25)
    else:
        xticks_Mw = np.arange(0.75, 3.5, 0.25)

    plots_path = os.path.join(results_plots_path, namedf, 'alltargets_v2')
    os.makedirs(plots_path, exist_ok=True)
    fig, axs = plt.subplots(
        len(target_names), len(which_methods), figsize=(15, 18),
        sharey='row', sharex='col')

    imethod, method = list(enumerate(which_methods))[0]
    for imethod, method in enumerate(which_methods):
        method2 = which_methods_names[method]
        print(f'Considering name: {namedf}, method: {method}')
        dff_features = dfs[namedf][
            features_cols + target_names + ['event']
            + list(s_targets.values())]

        target = target_names[0]

        if not y_range:
            lims_resid_all = max(lims_resid.values(), key=lambda v: v[1])
        else:
            lims_resid_all = y_range

        for itarget, target in enumerate(target_names):
            log_target = f'log10({target})'
            pred_log_target = f'{log_target}_pred'
            resid_log_target = f'{log_target}_resid'
            pred_target = f'{target}_pred'
            s_target = s_targets[target]
            if namedf == 'overall':
                y_true = pd.concat([pd.read_pickle(os.path.join(
                    results_comparison_path, f'y_true_{n}_{target}.pkl'))
                    for n in ['trainval', 'test']])
                ys_pred = pd.concat([pd.read_pickle(os.path.join(
                    results_comparison_path,
                    f'y_pred_{n}_{target}_{method}.pkl'))
                    for n in ['trainval', 'test']])
            else:
                y_true: pd.Series = pd.read_pickle(os.path.join(
                    results_comparison_path, f'y_true_{namedf}_{target}.pkl'))
                ys_pred = pd.read_pickle(os.path.join(
                    results_comparison_path,
                    f'y_pred_{namedf}_{target}_{method}.pkl'))

            y_pred_mean = ys_pred.mean(axis=1).rename(pred_log_target)

            dff = pd.concat(
                [dff_features, y_true, ys_pred, y_pred_mean], axis=1)

            dff[resid_log_target] = dff[log_target] - dff[pred_log_target]
            dff[pred_target] = np.power(10, dff[pred_log_target])

            # Grouped means and stds for Mw
            means_Mw = dff[[resid_log_target, 'Mw']].groupby('Mw').agg(avg_aggreg)
            stds_Mw = dff[[resid_log_target, 'Mw']].groupby('Mw').std()
            min_stds_Mw = means_Mw - stds_Mw
            max_stds_Mw = means_Mw + stds_Mw
            min_lines_Mw = means_Mw.index.values - 0.01 * xticks_Mw_range
            max_lines_Mw = means_Mw.index.values + 0.01 * xticks_Mw_range

            ax = axs[itarget, imethod]
            # Plot distribution Residual with Mw, with grouped means and stds
            sns.scatterplot(
                data=dff, x='Mw', y=resid_log_target, ax=ax, alpha=0.4,
                edgecolor='none', s=80, zorder=0)
            ax.vlines(
                means_Mw.index.values - 0.001, ymin=min_stds_Mw,
                ymax=max_stds_Mw, colors='black', linewidths=2.5,
                edgecolor='black', zorder=1)
            for i in range(len(min_stds_Mw)):
                ax.hlines(
                    [min_stds_Mw.iloc[i].values[0],
                     max_stds_Mw.iloc[i].values[0]],
                    xmin=min_lines_Mw[i], xmax=max_lines_Mw[i],
                    colors='black', zorder=1)
            ax.axhline(0, linestyle='--', c='black', zorder=1)
            sns.scatterplot(
                x=means_Mw.index.values, y=means_Mw.iloc[:, 0].values,
                color=red_color, edgecolor='none', s=60, ax=ax, zorder=2)
            if itarget == 0:
                ax.set_title(f'{method2}')
            ax.text(0.01, 0.9, target_labels[target], transform=ax.transAxes)
            ax.set_xticks(xticks_Mw)
            ax.set_ylim(lims_resid_all)
            ax.set(xlabel=r'Magnitude ($M_w$)', ylabel='Residuals')

    # fig.suptitle(f'Method: {method2}')
    fig.tight_layout()
    if save_plot:
        fig.savefig(os.path.join(
            plots_path, f'resid-grouped_Mw{y_range_s}_bin={n_bins}_avg_aggreg={avg_aggreg}.png'), **savefig_kwargs)
    if show_fig:
        plt.show()
    plt.close()

# %%
# Plot residuals unified

methods_kwargs = {
    'MOD3': dict(marker='o', edgecolor='none', color='gray', s=40),
    # 'deeplearn': dict(marker='x', color='black', s=80),
    'deeplearn': dict(marker='x', color='black', s=80)}

namedf = list(namedfs)[0]
for namedf in namedfs:

    if namedf == 'trainval':
        xticks_Mw = np.arange(1.25, 3.5, 0.25)
    else:
        xticks_Mw = np.arange(0.75, 3.5, 0.25)
    xticks_Rhypo = np.arange(0, 21, 2)

    fig, axs = plt.subplots(
        len(target_names), 2, figsize=(15, 18), sharey='row', sharex='col')
    dff_features = dfs[namedf][
        features_cols + target_names + ['event'] + list(s_targets.values())]

    plots_path = os.path.join(results_plots_path, namedf, 'alltargets')
    os.makedirs(plots_path, exist_ok=True)
    target = target_names[0]
    lims_resid_all = max(lims_resid.values(), key=lambda v: v[1])
    for itarget, target in enumerate(target_names):
        log_target = f'log10({target})'
        pred_log_target = f'{log_target}_pred'
        resid_log_target = f'{log_target}_resid'
        pred_target = f'{target}_pred'
        s_target = s_targets[target]
        ax = axs[itarget, 0]
        for method, method_kwargs in methods_kwargs.items():
            if namedf == 'overall':
                y_true = pd.concat([pd.read_pickle(os.path.join(
                    results_comparison_path, f'y_true_{n}_{target}.pkl'))
                    for n in ['trainval', 'test']])
                ys_pred = pd.concat([pd.read_pickle(os.path.join(
                    results_comparison_path,
                    f'y_pred_{n}_{target}_{method}.pkl'))
                    for n in ['trainval', 'test']])
            else:
                y_true: pd.Series = pd.read_pickle(os.path.join(
                    results_comparison_path, f'y_true_{namedf}_{target}.pkl'))
                ys_pred = pd.read_pickle(os.path.join(
                    results_comparison_path,
                    f'y_pred_{namedf}_{target}_{method}.pkl'))

            y_pred_mean = ys_pred.mean(axis=1).rename(pred_log_target)

            dff = pd.concat([dff_features, y_true, ys_pred, y_pred_mean], axis=1)

            dff[resid_log_target] = dff[log_target] - dff[pred_log_target]
            dff[pred_target] = np.power(10, dff[pred_log_target])

            method2 = which_methods_names[method]
            sns.scatterplot(
                data=dff, x='Rhypo', y=resid_log_target, ax=ax, alpha=1,
                **method_kwargs)
        ax.text(0.99, 0.9, target_labels[target], horizontalalignment='right',
                transform=ax.transAxes)
        ax.axhline(0, linestyle='--', c='black')
        ax.set_ylim(lims_resid_all)
        ax.set_xticks(xticks_Rhypo)
        ax.set(xlabel='Hypocentral distance (km)', ylabel='Residuals')

        ax = axs[itarget, 1]
        for method, method_kwargs in methods_kwargs.items():
            if namedf == 'overall':
                y_true = pd.concat([pd.read_pickle(os.path.join(
                    results_comparison_path, f'y_true_{n}_{target}.pkl'))
                    for n in ['trainval', 'test']])
                ys_pred = pd.concat([pd.read_pickle(os.path.join(
                    results_comparison_path,
                    f'y_pred_{n}_{target}_{method}.pkl'))
                    for n in ['trainval', 'test']])
            else:
                y_true: pd.Series = pd.read_pickle(os.path.join(
                    results_comparison_path, f'y_true_{namedf}_{target}.pkl'))
                ys_pred = pd.read_pickle(os.path.join(
                    results_comparison_path,
                    f'y_pred_{namedf}_{target}_{method}.pkl'))

            y_pred_mean = ys_pred.mean(axis=1).rename(pred_log_target)

            dff = pd.concat(
                [dff_features, y_true, ys_pred, y_pred_mean], axis=1)

            dff[resid_log_target] = dff[log_target] - dff[pred_log_target]
            dff[pred_target] = np.power(10, dff[pred_log_target])

            method2 = which_methods_names[method]
            sns.scatterplot(
                data=dff, x='Mw', y=resid_log_target, ax=ax, alpha=1,
                **method_kwargs)
        ax.text(0.01, 0.9, target_labels[target], transform=ax.transAxes)
        ax.axhline(0, linestyle='--', c='black')
        ax.set_xticks(xticks_Mw)
        ax.set_ylim(lims_resid_all)
        ax.set(xlabel=r'Magnitude ($M_w$)', ylabel='Residuals')
    fig.tight_layout()
    if save_plot:
        methods_save = '_vs_'.join(list(methods_kwargs.keys()))
        fig.savefig(os.path.join(
            plots_path, f'resids-improvements_{methods_save}.png'),
            **savefig_kwargs)
    if show_fig:
        plt.show()
    plt.close()

# %%
# Plot

modname_styles = {'MOD3': ['--'], 'deeplearn': ['-']}
colors = {'MOD3': 'black', 'deeplearn': 'red'}

Mw_vals = [1.5, 2.0, 3.0]
s_vals = [0]

Mw_range_dict = {
    Mw_vals[0]: dict(
        Mws=[Mw_vals[0]], min=-np.Inf, max=1.5, l_comp=pd.Series.gt,
        r_comp=pd.Series.le),
    Mw_vals[1]: dict(
        Mws=[Mw_vals[1]], min=1.5, max=2.5, l_comp=pd.Series.gt,
        r_comp=pd.Series.lt),
    Mw_vals[2]: dict(
        Mws=[Mw_vals[2]], min=2.5, max=+np.Inf, l_comp=pd.Series.ge, r_comp=pd.Series.lt)}


Rhypo_ticks = [
    x * (10**n) for n in range(-1, 2, 1) for x in [1, 2, 5]] + [100]

modnames = ['MOD3', 'deeplearn']

preds_plot = {modname: pd.read_pickle(os.path.join(
    results_comparison_path, f'df_preds_plot_{modname}.pkl'))
    for modname in modnames}

# %%

namedf_map = {
    'overall': 'original and test dataset',
    'trainval': 'original dataset',
    'test': 'test dataset'
}

target_plot_groups = [target_names]

show_single = False
with_density = False
legend_each_subplot = True

cutoff_y_pow_interval = None
cutoff_y_pow_interval = [-6, 1]


# R1 addition
def get_Mw_range_str(Mw):
    Mw_range = Mw_range_dict[Mw]
    if Mw_range['l_comp'] == pd.Series.gt:
        l_comp_str = '>'
    elif Mw_range['l_comp'] == pd.Series.ge:
        l_comp_str = '≥'
    else:
        assert False

    if Mw_range['r_comp'] == pd.Series.lt:
        r_comp_str = '<'
    elif Mw_range['r_comp'] == pd.Series.le:
        r_comp_str = '≤'
    else:
        assert False

    Mw_min = Mw_range['min']
    Mw_max = Mw_range['max']

    if np.isinf(Mw_range['min']):
        return f'Mw {r_comp_str} {Mw_max}'
    elif np.isinf(Mw_range['max']):
        return f'Mw {l_comp_str} {Mw_min}'
    else:
        l_comp_str = '<' if l_comp_str == '>' else '≤'
        return f'{Mw_min} {l_comp_str} Mw {r_comp_str} {Mw_max}'


text_above = False
text_above = True

density_s = ''
if with_density:
    density_s = '_density'
namedf = next(iter(namedfs))
for namedf in namedfs:
    dff = dfs[namedf]
    s = s_vals[0]
    for s in s_vals:
        plots_path = os.path.join(results_plots_path, namedf, 'alltargets_v2')
        os.makedirs(plots_path, exist_ok=True)
        for target_plot_group in target_plot_groups:
            if show_single is False:
                if with_density:
                    figsize = (15, 20)
                else:
                    figsize = (15, 20)
                fig, axs = plt.subplots(
                    len(target_plot_group), len(Mw_vals), sharex=True,
                    sharey='row', figsize=figsize)
            for i, target in enumerate(target_plot_group):
                for j, Mw in enumerate(Mw_vals):
                    if not show_single:
                        ax = axs[i, j]
                    df_plot = dff[['Rhypo', 'Mw', target]].copy()
                    Mw_min = Mw_range_dict[Mw]['min']
                    Mw_max = Mw_range_dict[Mw]['max']
                    Mw_l_comp = Mw_range_dict[Mw]['l_comp']
                    Mw_r_comp = Mw_range_dict[Mw]['r_comp']
                    Mw_Mws = Mw_range_dict[Mw]['Mws']

                    df_plot = df_plot[Mw_l_comp(df_plot['Mw'], Mw_min)
                                      & Mw_r_comp(df_plot['Mw'], Mw_max)]

                    if show_single:
                        if with_density:
                            figsize = (9, 6)
                        else:
                            figsize = (7, 6)
                        fig, ax = plt.subplots(figsize=figsize)
                    if with_density:
                        values_kde_target = np.log10(
                            df_plot[[target, 'Rhypo']].values.T.copy())
                        kernel_target = stats.gaussian_kde(
                            values_kde_target)(values_kde_target)

                        idx = np.arange(len(df_plot))
                        # idx = kernel_target.argsort()
                        kernel_range_target = [
                            kernel_target.min(), kernel_target.max()]
                        norm_kernel_target = plt.Normalize(
                            *kernel_range_target)
                        sm_kernel_target = plt.cm.ScalarMappable(
                            cmap='viridis', norm=norm_kernel_target)
                        sm_kernel_target.set_array([])
                    else:
                        idx = np.arange(len(df_plot))

                    for modname in modnames:
                        df_preds_plot_mod = preds_plot[modname]
                        for i_Mw, Mw_show in enumerate(Mw_Mws):
                            pred_colname = f'Mw={Mw_show},s={s},target={target}'
                            log10_y_pred_plot = df_preds_plot_mod[pred_colname]
                            y_pred_plot = np.power(10., log10_y_pred_plot)
                            ax.plot(df_preds_plot_mod['Rhypo'], y_pred_plot,
                                    # label=f'{modname}, Mw={Mw_show}',
                                    label=which_methods_names[modname],
                                    linestyle=modname_styles[modname][i_Mw],
                                    color='black')
                    if with_density:
                        sns.scatterplot(
                            data=df_plot.iloc[idx], x='Rhypo', y=target, ax=ax,
                            alpha=0.33, c=kernel_target, s=10,
                            cmap=sm_kernel_target.cmap, edgecolor='none')
                    else:
                        sns.scatterplot(
                            data=df_plot.iloc[idx], x='Rhypo', y=target, ax=ax,
                            s=10, alpha=0.33, color='gray', edgecolor='none')
                    if not show_single and j == axs.shape[1] - 1:
                        ax.set(xlabel='Rhypo', xlim=[0.5, 20],
                               xticks=[0.5, 1, 2, 5, 10, 20])
                    cutoff_y_pow_interval_ = cutoff_y_pow_interval
                    if cutoff_y_pow_interval is None:
                        cutoff_y_pow_interval_ = [-8, 1]
                        # ax.set(ylim=[1e-8, 10], yticks=10.**np.arange(-8, 2))
                    ylim = list(map(
                        lambda x: pow(10, x), cutoff_y_pow_interval))
                    ypow_range = (
                        cutoff_y_pow_interval[0], cutoff_y_pow_interval[1] + 1)
                    ax.set(ylim=ylim, yticks=10.**np.arange(*ypow_range))
                    ax.set(xlabel='Hypocentral distance (km)',
                           ylabel=target_labels[target])
                    if not show_single and j != 0:
                        # plt.setp(ax.get_xticklabels(), visible=False)
                        plt.setp(ax.get_yticklabels(), visible=False)
                    ax.set(xscale='log', yscale='log')
                    if not text_above or j in [1, 2]:
                        ax.legend(loc='lower left')
                    else:
                        ax.legend(loc='upper left')

                    if not text_above:
                        ax.text(
                            0.95, 0.05, get_Mw_range_str(Mw), horizontalalignment='right',
                            transform=ax.transAxes)
                    else:
                        ax.text(
                            0.95, 0.80, get_Mw_range_str(Mw), horizontalalignment='right',
                            transform=ax.transAxes)

                    if with_density and legend_each_subplot:
                        cbar = fig.colorbar(
                            sm_kernel_target, ticks=kernel_range_target,
                            label='Density', ax=ax)
                        cbar.ax.set_yticklabels(['Low', 'High'])
                        cbar.ax.set_ylabel(cbar.ax.get_ylabel(), labelpad=-40)

                    if show_single:
                        fig.suptitle(f'{namedf_map[namedf]}, target: {target}')
                        if save_plot:
                            fig.savefig(os.path.join(
                                plots_path, f'distr_plot{density_s}_{namedf}'
                                f'_{target}_Mw={Mw}.png'),
                                **savefig_kwargs)
                        if show_fig:
                            plt.show()
                        plt.close()
            if not show_single:
                # fig.suptitle(f'{namedf_map[namedf]}')
                if with_density and not legend_each_subplot:
                    # fig.subplots_adjust(right=0.9)
                    fig.tight_layout()
                    # fig.canvas.draw()
                    cax = fig.add_axes([
                        axs[-1, -1].get_position().x1 + 0.01,
                        axs[-1, -1].get_position().y0,
                        0.02,
                        axs[0, -1].get_position().y1 - axs[-1, -1].get_position().y0])
                    cbar = fig.colorbar(
                        sm_kernel_target, ticks=kernel_range_target,
                        label='Density', cax=cax, use_gridspec=True, ax=axs.ravel().tolist())
                    cbar.ax.set_yticklabels(['Low', 'High'])
                    # cbar.ax.set_ylabel(cbar.ax.get_ylabel(), labelpad=-30)
                else:
                    fig.tight_layout()
                if save_plot:
                    fig.savefig(os.path.join(
                        plots_path, f'distr_plot{density_s}_{namedf}_{modnames[1]}_new.png'),
                        **savefig_kwargs)
                if show_fig:
                    plt.show()
                plt.close()

# %%

comparison_results_file = os.path.join(
    results_comparison_path, 'comparison_results.txt')

ff = sys.stdout
if True:  # with open(comparison_results_file, 'w') as ff:
    print('Comparison results\n\n', file=ff)

namedf = 'overall'
dff = dfs[namedf]

for namedf, dff in dfs.items():
    if True:  # with open(comparison_results_file, 'a') as ff:
        print(f'Dataset: {namedf}\n', file=ff)

    for method in which_methods:
        scores = ['sigma_inter', 'sigma_intra', 'sigma', 'mae', 'R^2', 'Nobs']
        df_results = pd.DataFrame(
            np.nan, index=target_names, columns=scores, dtype=str)

        method2 = which_methods_names[method]
        for target in target_names:
            if namedf == 'overall':
                y_true = pd.concat([pd.read_pickle(os.path.join(
                    results_comparison_path, f'y_true_{n}_{target}.pkl'))
                    for n in ['trainval', 'test']])
                ys_pred = pd.concat([pd.read_pickle(os.path.join(
                    results_comparison_path,
                    f'y_pred_{n}_{target}_{method}.pkl'))
                    for n in ['trainval', 'test']])
            else:
                y_true: pd.Series = pd.read_pickle(os.path.join(
                    results_comparison_path, f'y_true_{namedf}_{target}.pkl'))
                ys_pred = pd.read_pickle(os.path.join(
                    results_comparison_path,
                    f'y_pred_{namedf}_{target}_{method}.pkl'))

            log_target = f'log10({target})'
            pred_log_target = f'{log_target}_pred'
            resid_log_target = f'{log_target}_resid'
            # y_pred = ys_pred.mean(axis=1).rename(pred_log_target)

            group = dff['event'].astype('int')

            r2s_score = [
                skmetrics.r2_score(y_true, ys_pred[col])
                for col in ys_pred.columns]

            maes_score = [
                skmetrics.mean_absolute_error(y_true, ys_pred[col])
                for col in ys_pred.columns]

            sigmas_inter = [between_sigma(y_true, ys_pred[col], group)
                            for col in ys_pred.columns]
            sigmas_intra = [within_sigma(y_true, ys_pred[col], group)
                            for col in ys_pred.columns]
            sigmas = [total_sigma(y_true, ys_pred[col])
                      for col in ys_pred.columns]

            df_results.loc[target, 'Nobs'] = len(y_true)
            if method == 'MOD1' or method == 'MOD3' or \
                    len(ys_pred.columns) == 1:
                df_results.loc[target, 'sigma_inter'] = \
                    f'{np.mean(sigmas_inter):.3f}'
                df_results.loc[target, 'sigma_intra'] = \
                    f'{np.mean(sigmas_intra):.3f}'
                df_results.loc[target, 'sigma'] = f'{np.mean(sigmas):.3f}'
                df_results.loc[target, 'R^2'] = f'{np.mean(r2s_score):.3f}'
                df_results.loc[target, 'mae'] = f'{np.mean(maes_score):.3f}'
            else:
                df_results.loc[target, 'sigma_inter'] = \
                    f'{np.mean(sigmas_inter):.3f} ± ' \
                    f'{np.std(sigmas_inter, ddof=1):.3f}'
                df_results.loc[target, 'sigma_intra'] = \
                    f'{np.mean(sigmas_intra):.3f} ± ' \
                    f'{np.std(sigmas_intra, ddof=1):.3f}'
                df_results.loc[target, 'sigma'] = \
                    f'{np.mean(sigmas):.3f} ± {np.std(sigmas, ddof=1):.3f}'
                df_results.loc[target, 'R^2'] = \
                    f'{np.mean(r2s_score):.3f} ± ' \
                    f'{np.std(r2s_score, ddof=1):.3f}'
                df_results.loc[target, 'mae'] = \
                    f'{np.mean(maes_score):.3f} ± ' \
                    f'{np.std(maes_score, ddof=1):.3f}'

        if True:  # with open(comparison_results_file, 'a') as ff:
            print(f'Method: {method2}', file=ff)
            print(df_results, file=ff)
            print('', file=ff)

# %%
