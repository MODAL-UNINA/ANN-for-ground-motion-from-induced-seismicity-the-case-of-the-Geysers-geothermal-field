#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %%
# Imports

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from utils.yamlhandler import YamlHandler  # Configuration
from utils.args import getparam

from utils import torchutils

from sklearn.preprocessing import \
    FunctionTransformer, MinMaxScaler, PowerTransformer, StandardScaler
from sklearn.compose import make_column_transformer

from sklearn.model_selection import GroupShuffleSplit

from sklearn import metrics as skmetrics

from fastai.tabular.all import *
from fastai.callback.all import EarlyStoppingCallback
from utils.fastaiutils import *

from utils.metrics import total_sigma, within_sigma, between_sigma

import os
import time
import sys

pd.set_option('display.precision', 3)
pd.set_option('display.width', 200)

plt.style.use('seaborn')

# %%
# Main Paths

cur_dir = os.getcwd()
parent_dir = os.path.dirname(cur_dir)

# %%
# GPU ID

gpu_id_s = '-1'
gpu_id = int(os.environ.get('CUDA_VISIBLE_DEVICES', gpu_id_s))

# %%

which_region = getparam('region', 'geysers')
config_name = getparam('config', 'deeplearn')

# %%
# Target

target_ = getparam('use_target', 'none', optional=True)

# %%
# Show the plot of the loss

do_showplot = bool(getparam('show_plot', 1, optional=True))

# %%
# Save the model

do_savemodel = bool(getparam('save_model', 1, optional=True))

# %%
# Save the prediction if needed

do_savepred = bool(getparam('save_pred', 1, optional=True))

# %%
# Show overall and/or test score

do_show_overall = bool(getparam('show_overall', 1, optional=True))
do_show_test = bool(getparam('show_test', 1, optional=True))

# %%
# Derived paths from the config name

results_path = os.path.join(parent_dir, 'results', config_name)
os.makedirs(results_path, exist_ok=True)

# %%
# Load of the configuration

config = dict2obj(YamlHandler.safe_load_file(
    os.path.join(parent_dir, 'config', 'training', 'deeplearn.yml')))

print('Current configuration:')
print(config)

# %%
# Compatibility checks.

early_terminate = False
early_terminate = early_terminate or ((
    config.neurons_l2 == 0 and config.neurons_l3 != 0)
    or (config.neurons_l3 == 0 and config.neurons_l4 != 0)
    or (config.neurons_l4 == 0 and config.neurons_l5 != 0)
    or (config.neurons_l5 == 0 and config.neurons_l6 != 0)
    or (config.neurons_l6 == 0 and config.neurons_l7 != 0))

if early_terminate:
    print('Neurons configuration unexpected.')
    sys.exit(0)

# %%
# Target

if target_ == 'none':
    target_ = None

if target_ is None:
    target_ = config.target

# %%
# Load of train data

df = pd.read_pickle(
    os.path.join(parent_dir, 'dataset', 'features', f'Xy_{which_region}_{config_name}_train.pkl.bz2'))
df_test = pd.read_pickle(
    os.path.join(parent_dir, 'dataset', 'features', f'Xy_{which_region}_{config_name}_test.pkl.bz2'))

# %%
# Load the test data if needed

comparison_path = None

if do_savepred:
    comparison_path = os.path.join(results_path, 'comparison')
    os.makedirs(comparison_path, exist_ok=True)

# %%
# Seed

seed = torchutils.seed_everything(config.seed)

# %%
# Device creation

device = torch.device(
    # f'cuda:{gpu_id}' if
    'cuda:0' if torch.cuda.is_available() and gpu_id != -1 else "cpu")
if device.type == 'cuda':
    torch.cuda.set_device(device)

# %%
# Features and targets

features_cols_nos = ['Mw', 'Rhypo']

targets = ['PGV', 'PGA', 'SA-0.2', 'SA-0.5', 'SA-1.0']

if target_ == 'all':
    l_targets = targets
else:
    l_targets = [target_]

group_col = 'event'

df[group_col] = df.index.get_level_values(group_col)
df_test[group_col] = df_test.index.get_level_values(group_col)

dfs = {'trainval': df, 'test': df_test}
preds = {target: {'trainval': [], 'test': []} for target in l_targets}

# %%
# Learner params init

learner_params = AttrDict()

# %%
# Model parameters

act_func = torchutils.get_act_module(config.act)

layers = L(config.neurons_l1)

if config.neurons_l2 != 0:
    layers.append(config.neurons_l2)
if config.neurons_l3 != 0:
    layers.append(config.neurons_l3)
if config.neurons_l4 != 0:
    layers.append(config.neurons_l4)
if config.neurons_l5 != 0:
    layers.append(config.neurons_l5)
if config.neurons_l6 != 0:
    layers.append(config.neurons_l6)
if config.neurons_l7 != 0:
    layers.append(config.neurons_l7)

learner_params.layers = layers

learner_params.config = dict(
    use_bn=config.bn, bn_final=False, bn_cont=config.bn,
    act_cls=act_func, lin_first=True, ps=config.pdrop)


# %%
# Train parameters

loss_func_dict = {
    'mse': nn.MSELoss(),
    'r2': R2Loss(),
    'sigma': SigmaLoss(),
    'msesigmatotal': MSESigmaTotalLoss(
        coeff_mse=config.coeff_mse, coeff_sigma=config.coeff_sigma_total),
    'msesigma': MSESigmaLoss(
        coeff_mse=config.coeff_mse,
        coeff_sigma_within=config.coeff_sigma_within,
        coeff_sigma_between=config.coeff_sigma_between),
}

learner_params.loss_func = loss_func_dict[config.loss]

opt_func_dict = {
    'sgd_decoupled': SGD,
    'sgd': partial(SGD, decouple_wd=False),
    'rmsprop_decoupled': RMSProp,
    'rmsprop': partial(RMSProp, decouple_wd=False),
    'adam_decoupled': Adam,
    'adam': partial(Adam, decouple_wd=False),
}

learner_params.opt_func = opt_func_dict[config.opt]

learner_params.wd = config.reg_l2


# %%
# Is the loss function based on the events?

group_loss = config.loss == 'msesigma'

# %%
# Metrics


def rmse2(inp, targ):
    return torch.sqrt(F.mse_loss(inp, targ))


def r2score2(targ, inp, **kwargs):
    if group_loss:
        targ_ = targ[:, 1:]
    else:
        targ_ = targ
    return skmetrics.r2_score(targ_, inp, **kwargs)


def R2Score(sample_weight=None, flatten=True):
    "R2 score between predictions and targets"
    return skm_to_fastai(
        r2score2, is_class=False, sample_weight=sample_weight,
        flatten=flatten)


rmse_metric = AccumMetric(
    partial(metric_group_handler, m=rmse2, do_group=group_loss),
    flatten=not group_loss)
mae_metric = AccumMetric(
    partial(metric_group_handler, m=mae, do_group=group_loss),
    flatten=not group_loss)
r2score_metric = R2Score(flatten=not group_loss)


metrics = [rmse_metric, mae_metric, r2score_metric]

learner_params.metrics = metrics

# %%
# Train-val split

perc_val = config.perc_val
if perc_val == 'copy_train':
    perc_val = 0

if config.stratify:
    stratify = df.index.get_level_values(group_col)
else:
    stratify = None

# %%
# Rescaling

scalerX = FunctionTransformer()

if config.rescaleX == 'standard':
    scalerX = StandardScaler()
elif config.rescaleX == 'minmax':
    scalerX = MinMaxScaler()
elif config.rescaleX == 'yeo-johnson':
    scalerX = PowerTransformer(method='yeo-johnson')
elif config.rescaleX == 'box-cox':
    scalerX = PowerTransformer(method='box-cox')

scalerY = FunctionTransformer()

if config.rescaleY == 'standard':
    scalerY = StandardScaler()
elif config.rescaleY == 'minmax':
    scalerY = MinMaxScaler()
elif config.rescaleY == 'yeo-johnson':
    scalerY = PowerTransformer(method='yeo-johnson')
elif config.rescaleY == 'box-cox':
    scalerY = PowerTransformer(method='box-cox')

scalerX = make_column_transformer(
    (scalerX, features_cols_nos), remainder='passthrough')

# %%
# Loss plot function


def get_df_metrics(recorder):
    values = np.stack(recorder.values)
    n_metrics = values.shape[1]
    names = recorder.metric_names[1:n_metrics + 1]

    return pd.DataFrame(values, columns=names)


def plot_loss(recorder, skip_start=5, with_valid=True):
    n_values = len(recorder.values)
    metrics2 = np.stack(recorder.values)
    n_metrics = metrics2.shape[1]
    perc = 0.5
    names = recorder.metric_names[1:n_metrics + 1]
    sel_idxs = int(round(n_values * perc))
    if sel_idxs >= 2:
        metrics2 = np.concatenate((metrics2[:, :2], metrics2), -1)
        names = names[:2] + names

    df_metrics = pd.DataFrame(metrics2[skip_start:], columns=names)

    fig, ax = plt.subplots()
    df_metrics['train_loss'].plot(ax=ax)
    if with_valid:
        df_metrics['valid_loss'].plot(ax=ax)
    return fig, ax


def plot_model_weights(learn):
    # TODO fix aspect ratios of the weights and bias plot
    paramiter = learn.parameters()
    while (weights := next(paramiter, None)) is not None:
        bias = next(paramiter)
        weights2 = weights.to('cpu').detach()
        bias2 = bias.to('cpu').detach().reshape(-1, 1)
        vmaxabs_weights = torch.abs(weights2).max()
        vmaxabs_bias = torch.abs(bias2).max()
        vmaxabs = np.max([vmaxabs_weights, vmaxabs_bias])
        fig, axs = plt.subplots(1, 2, sharey=True)
        axs[0].imshow(weights2, cmap='RdBu', vmin=-vmaxabs, vmax=vmaxabs)
        axs[1].imshow(bias2, cmap='RdBu', vmin=-vmaxabs, vmax=vmaxabs)
        # fig.colorbar()
        plt.show()


# %%


def get_scores(y_true, y_pred, group, prefix=''):
    result_k = pd.Series(dtype=float)

    result_k[f'{prefix}sigma_inter'] = between_sigma(y_true, y_pred, group)
    result_k[f'{prefix}sigma_intra'] = within_sigma(y_true, y_pred, group)
    result_k[f'{prefix}sigma'] = total_sigma(y_true, y_pred)
    result_k[f'{prefix}R2'] = skmetrics.r2_score(y_true, y_pred)
    result_k[f'{prefix}rmse'] = skmetrics.mean_squared_error(
        y_true, y_pred, squared=False)
    result_k[f'{prefix}mae'] = skmetrics.mean_absolute_error(y_true, y_pred)

    return result_k


# %%
# Bootstrap

indxs = range_of(df)

results_epochs_all = {}
results_valid_all = {}
results_overall_all = {}
results_test_all = {}
dfs_metrics_vals_all = {}

target = l_targets[0]
for target in l_targets:
    log_target = f'log10({target})'
    pred_log_target = f'{log_target}_pred'

    if group_loss:
        y_names = [group_col, log_target]
    else:
        y_names = log_target

    print(f'Running training on {log_target} ...')

    s_target = f's({target})'
    if config.ohe:
        s_target_ohes = [f'{s_target}_{j}' for j in range(2)]
        features_cols = features_cols_nos + s_target_ohes
    else:
        features_cols = features_cols_nos + [s_target]

    if config.ohe:
        df[s_target_ohes[0]] = (df[s_target] == -1).astype('float')
        df[s_target_ohes[1]] = (df[s_target] == 1).astype('float')
        df_test[s_target_ohes[0]] = (
            df_test[s_target] == -1).astype('float')
        df_test[s_target_ohes[1]] = (
            df_test[s_target] == 1).astype('float')

    # Callbacks
    model_fname = f'GMPE({target})'

    import datetime
    ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_fname += f'_{ts}'

    comp = None
    if any(x in config.monitor for x in ['loss', 'sigma']):
        comp = np.less
    if any(x in config.monitor for x in ['r2_score']):
        comp = np.greater

    learner_params.cbs = [SaveModelCallback(
        fname=model_fname, monitor=config.monitor, comp=comp)]

    if config.patience >= 0:
        learner_params.cbs.append(EarlyStoppingCallback(
            patience=config.patience, monitor=config.monitor, comp=comp))

    results_epochs = []
    results_valid = []
    results_overall = []
    results_test = []
    dfs_metrics_vals = []

    k = 0
    split_seed = config.split_seeds[k]
    for k, split_seed in enumerate(config.split_seeds):
        # Train-val split
        if config.stratify:
            splits = TrainTestSplitter(
                test_size=perc_val, random_state=split_seed,
                stratify=stratify)(indxs)
        elif config.grouped:
            grp_splt = GroupShuffleSplit(
                n_splits=1, test_size=perc_val, random_state=split_seed)
            splits = next(grp_splt.split(indxs, groups=df[group_col].values))
            splits = (L(*splits[0]), L(*splits[1]))
        else:
            splits = RandomSplitter(
                valid_pct=perc_val, seed=split_seed)(indxs)

        # Train-val split and creation of dataloaders
        if config.perc_val == 'copy_train':
            splits = (splits[0], splits[0])

        # Data scaling
        df_transf = df.copy()
        df_train = df_transf.iloc[splits[0]]
        df_val = df_transf.iloc[splits[1]]

        scalerX.fit(df_train[features_cols])
        scalerY.fit(df_train[[log_target]])

        df_transf[features_cols] = scalerX.transform(df_transf[features_cols])
        df_transf[[log_target]] = scalerY.transform(df_transf[[log_target]])
        if group_loss:
            df_transf[group_col] = df_transf[group_col].astype('float')

        bs = config.batch_size
        val_bs = bs
        if bs is None:
            bs = len(df_train)
            val_bs = len(df_val)

        to = TabularPandas(
            df_transf, procs=[], cont_names=features_cols, y_names=y_names,
            splits=splits)
        dls = to.dataloaders(bs=bs, val_bs=val_bs, device=device)

        # Learner
        learn = tabular_learner_GMPE(
            dls, **learner_params, model_dir=results_path,
            y_range=(-config.range_eps, 1 + config.range_eps))
        learn.cbs[1].train_metrics = True

        if config.kernel_init != 'default' or config.bias_init != 'default':
            for layer in learn.model.layers:
                if not isinstance(layer, LinBnDrop):
                    continue
                for sublayer in layer:
                    if isinstance(sublayer, nn.Linear):
                        if config.kernel_init == 'glorot_uniform':
                            nn.init.xavier_uniform_(sublayer.weight)
                        if config.kernel_init == 'glorot_normal':
                            nn.init.xavier_normal_(sublayer.weight)
                        if config.kernel_init == 'he_uniform':
                            nn.init.kaiming_uniform_(sublayer.weight)
                        if config.kernel_init == 'he_normal':
                            nn.init.kaiming_normal_(sublayer.weight)
                        if config.kernel_init == 'orthogonal':
                            nn.init.orthogonal_(sublayer.weight)
                        if config.kernel_init == 'zeros':
                            nn.init.zeros_(sublayer.weight)
                        if config.kernel_init == 'ones':
                            nn.init.ones_(sublayer.weight)
                        if config.kernel_init == 'eye':
                            nn.init.eye_(sublayer.weight)
                        if config.bias_init == 'zeros':
                            nn.init.zeros_(sublayer.bias)
                        if config.bias_init == 'ones':
                            nn.init.ones_(sublayer.bias)

        if k == 0 and config.patience < 0:
            print(learn.summary())

        # Learning rate finder, if required
        lr = config.lr
        if lr is None:
            print('- Finding the best lr ...', end='')
            with ContextManagers([learn.no_logging(), learn.no_bar()]):
                lr_min, lr_steep, lr_valley, lr_slide = learn.lr_find(
                    suggest_funcs=(minimum, steep, valley, slide), num_it=200)
            print(f"Minimum/10:\t{lr_min:.2e}")
            print(f"Steepest point:\t{lr_steep:.2e}")
            print(f"Longest valley:\t{lr_valley:.2e}")
            print(f"Slide interval:\t{lr_slide:.2e}")
            opt_lr = max(lr_min, lr_steep, lr_valley, lr_slide)
            lr = 10. ** int(np.floor(np.log10(opt_lr)))
            print(f' done! Value: {lr}')

        # Model training

        lr_new = lr
        print('- Training the model ...')
        tk0 = time.time()
        reset_opt = True
        reset_on_fit = True
        best = None
        best_epoch = 0
        metrics_vals = []
        with ContextManagers([learn.no_logging(), learn.no_bar()]):
            while lr_new >= lr:  # / 1000:
                if config.patience >= 0:
                    assert isinstance(learn.cbs[-1], EarlyStoppingCallback)
                    assert isinstance(learn.cbs[-2], SaveModelCallback)
                    learn.cbs[-1].reset_on_fit = reset_on_fit
                    learn.cbs[-2].reset_on_fit = reset_on_fit
                    learn.cbs[-1].best = best
                    learn.cbs[-2].best = best
                else:
                    assert isinstance(learn.cbs[-1], SaveModelCallback)
                    learn.cbs[-1].reset_on_fit = reset_on_fit
                    learn.cbs[-1].best = best

                if config.fit_func == 'normal':
                    learn.fit(config.epochs, lr=lr_new, reset_opt=reset_opt)
                elif config.fit_func == 'one_cycle':
                    # pct_start = (5000 * 0.25 / config.epochs)
                    learn.fit_one_cycle(
                        config.epochs, lr_max=lr_new, reset_opt=reset_opt,
                        pct_start=config.cycle_perc_start)
                else:
                    raise ValueError('Bad config fit_func!')
                lr_new /= 10
                reset_opt = False
                reset_on_fit = False
                best = learn.cbs[-1].best

                best_epoch = learn.cbs[-1].epoch - learn.cbs[-1].wait
                print(f'Done cycle with lr = {lr_new}. Best value: {best}')

                metrics_vals.append(get_df_metrics(learn.recorder))
        tkf = time.time()
        print(f'Done! Execution time: {tkf - tk0:.2f} secs.')

        df_metrics_vals = pd.concat(metrics_vals, ignore_index=True)
        df_metrics_vals['Train loss'] = df_metrics_vals['train_loss']
        df_metrics_vals['Test loss'] = df_metrics_vals['valid_loss']

        dfs_metrics_vals.append(df_metrics_vals)

        if do_savemodel:
            # Export learner
            export_file = os.path.join(
                results_path, f'GMPE_v2_1({target})_k={k}.pkl')
            learn.export(fname=export_file)

        del learn.cbs[-2]

        df_y_val = df_transf.loc[
            dls.valid.dataset.items.index, [log_target, group_col]].copy()

        with ContextManagers([learn.no_logging(), learn.no_bar()]):
            df_y_val[pred_log_target] = \
                learn.get_preds(dl=dls.valid)[0].numpy()

        df_y_val[[log_target, pred_log_target]] = scalerY.inverse_transform(
            df_y_val[[log_target, pred_log_target]])

        result_k = get_scores(
            df_y_val[log_target], df_y_val[pred_log_target],
            df_y_val[group_col], prefix='valid_')
        results_valid.append(result_k)
        print(f'Results of run {k} (valid):')
        print(pd.DataFrame(result_k).T)

        result_k_epochs = pd.Series(dtype='int')
        result_k_epochs['epochs'] = len(df_metrics_vals)
        result_k_epochs['best_epoch'] = df_metrics_vals[
            df_metrics_vals[config.monitor] == best].index[0] + 1
        result_k_epochs[f'best_{config.monitor}'] = best
        result_k_epochs['time'] = tkf - tk0
        results_epochs.append(result_k_epochs)

        result_run = {}
        for namedf, do_show, dff in zip(
                ['overall', 'test'], [do_show_overall, do_show_test],
                [df, df_test]):
            if not do_show:
                continue

            dff_transf = dff.copy()
            dff_transf[features_cols] = scalerX.transform(
                dff_transf[features_cols])
            dff_transf[log_target] = scalerY.transform(
                dff_transf[[log_target]]).flatten()
            if group_loss:
                dff_transf[group_col] = dff_transf[group_col].astype('float')

            dl_dff = learn.dls.test_dl(dff_transf)

            dff_y = dff_transf[[log_target, group_col]].copy()
            with ContextManagers([learn.no_logging(), learn.no_bar()]):
                dff_y[pred_log_target] = learn.get_preds(
                    dl=dl_dff)[0].numpy()

            dff_y[[log_target, pred_log_target]] = \
                scalerY.inverse_transform(
                    dff_y[[log_target, pred_log_target]])
            if group_loss:
                dff_y[group_col] = dff_y[group_col].astype('int')

            result_k_dff = get_scores(
                dff_y[log_target], dff_y[pred_log_target],
                dff_y[group_col], prefix=f'{namedf}_')
            if namedf == 'overall':
                results_overall.append(result_k_dff)
            if namedf == 'test':
                results_test.append(result_k_dff)
            result_run[namedf] = get_scores(
                dff_y[log_target], dff_y[pred_log_target],
                dff_y[group_col])

        print(f'Results of run {k}:')
        print(pd.DataFrame(result_run))

        # Save the predictions
        dff = df_test
        k_pred_log_target = f'{pred_log_target}_{k}'
        for namedff, dff in dfs.items():
            dff_transf = dff.copy()
            dff_transf[features_cols] = scalerX.transform(
                dff_transf[features_cols])
            dff_transf[log_target] = scalerY.transform(
                dff_transf[[log_target]]).flatten()
            if group_loss:
                dff_transf[group_col] = dff_transf[group_col].astype('float')

            dl_dff = learn.dls.test_dl(dff_transf)

            dff_target = dff_transf[[log_target]].copy()
            with ContextManagers([learn.no_logging(), learn.no_bar()]):
                dff_target[k_pred_log_target] = learn.get_preds(
                    dl=dl_dff)[0].numpy()

            dff_target[[log_target, k_pred_log_target]] = \
                scalerY.inverse_transform(
                    dff_target[[log_target, k_pred_log_target]])

            preds[target][namedff].append(dff_target[k_pred_log_target])

    results_epochs_all[target] = results_epochs
    results_valid_all[target] = results_valid
    results_overall_all[target] = results_overall
    results_test_all[target] = results_test
    dfs_metrics_vals_all[target] = dfs_metrics_vals
    # Get the results

    for nameres, ress in zip(
            ['epochs', 'valid', 'overall', 'test'],
            [results_epochs, results_valid, results_overall, results_test]):

        if len(ress) == 0:
            continue
        df_results = pd.concat(ress, axis=1).T

        print(f'Final results for target {target} ({nameres}):')
        print(df_results.T)

        if nameres == 'epochs':
            total_time = df_results['time'].sum()
            print(f'Total time: {total_time}')
            # print(f'Best {config.monitor}: {best}')
            continue

        all_metrics_names = list(df_results.columns)

        if len(df_results) > 1:
            print(f'Average results ({nameres}) for target {target}:')
            print(df_results.apply(['mean', 'std']).T.apply(
                lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}", axis=1))

            if nameres == 'valid':
                mean_results = df_results.mean()
                mean_results.index = [f'mean_{f}' for f in all_metrics_names]
                std_results = df_results.std()
                std_results.index = [f'std_{f}' for f in all_metrics_names]

                all_results = pd.concat([mean_results, std_results])

    # Save the predictions (and also the true values) on disk if needed

    if do_savepred:
        df_preds = {k: pd.concat(v, axis=1) for k, v in preds[target].items()}
        for namedff in dfs.keys():
            y_true = dfs[namedff][log_target]

            ys_pred = df_preds[namedff]

            filepath_true = os.path.join(
                comparison_path, f'y_true_{namedff}_{target}.pkl')
            if not os.path.isfile(filepath_true):
                y_true.to_pickle(filepath_true, protocol=-1)
            ys_pred.to_pickle(os.path.join(
                comparison_path, f'y_pred_{namedff}_{target}_{config_name}.pkl'),
                protocol=-1)
