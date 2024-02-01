#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 16:09:39 2021

@author: modal
"""

# %%
# Imports

from fastai.tabular.all import *
import random
import numpy as np

from typing import Optional


# %%
# From https://raw.githubusercontent.com/PyTorchLightning/pytorch-lightning/568a1e0a68fb8a2d8f2be7fae4f79c9068ca56fb/pytorch_lightning/utilities/seed.py

def _select_seed_randomly(
        min_seed_value: int = 0, max_seed_value: int = 255) -> int:
    return random.randint(min_seed_value, max_seed_value)


def seed_everything(seed: Optional[int] = None) -> int:
    """Function that sets seed for pseudo-random number generators in: pytorch,
    numpy, python.random .

    Args:
        seed: the integer value seed for global random state in Lightning.
            If `None`, will read seed from `PL_GLOBAL_SEED` env variable
            or select it randomly.
    """
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    if seed is None:
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    elif not isinstance(seed, int):
        seed = int(seed)

    if not (min_seed_value <= seed <= max_seed_value):
        print(f"{seed} is not in bounds, numpy accepts from "
              f"{min_seed_value} to {max_seed_value}")
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    # using `log.info` instead of `rank_zero_info`,
    # so users can verify the seed is properly set in distributed training.
    print(f"Global seed set to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return seed

# %%


def get_act_module(act, **kwargs):
    """
        Helper function to get the activation module from the string.
    """

    if act is None:
        return nn.Sigmoid(**kwargs)
    if isinstance(act, Module):
        return act

    assert isinstance(act, str), \
        'Activation must be either None, `nn.Module`, or `str`.'

    act_dict = {
        'relu': nn.ReLU,
        'sigmoid': nn.Sigmoid,
        'tanh': nn.Tanh,
    }

    assert act in act_dict.keys(), \
        f'Activation function not found: `{act}`.'

    return act_dict[act](**kwargs)


# %%


@patch
@delegates(subplots)
def plot_metrics(
        self: Recorder, nrows=None, ncols=None, figsize=None,
        final_losses=True, perc=.5, logy=True, **kwargs):
    n_values = len(self.recorder.values)
    if n_values < 2:
        print('not enough values to plot a chart')
        return
    metrics = np.stack(self.values)
    n_metrics = metrics.shape[1]
    names = self.metric_names[1:n_metrics + 1]
    if final_losses:
        sel_idxs = int(round(n_values * perc))
        if sel_idxs >= 2:
            metrics = np.concatenate((metrics[:, :2], metrics), -1)
            names = names[:2] + names
        else:
            final_losses = False
    n = len(names) - 1 - final_losses
    if nrows is None and ncols is None:
        nrows = int(math.sqrt(n))
        ncols = int(np.ceil(n / nrows))
    elif nrows is None:
        nrows = int(np.ceil(n / ncols))
    elif ncols is None:
        ncols = int(np.ceil(n / nrows))
    figsize = figsize or (ncols * 6, nrows * 4)
    fig, axs = subplots(nrows, ncols, figsize=figsize, **kwargs)
    axs = [ax if i < n else ax.set_axis_off()
           for i, ax in enumerate(axs.flatten())][:n]
    axs = ([axs[0]] * 2 + [axs[1]] * 2 + axs[2:]) \
        if final_losses else ([axs[0]] * 2 + axs[1:])
    for i, (name, ax) in enumerate(zip(names, axs)):
        if i in [0, 1]:
            ax.plot(metrics[:, i], color='#1f77b4' if i == 0 else '#ff7f0e',
                    label='valid' if i == 1 else 'train')
            ax.set_title('losses')
            ax.set_xlim(0, len(metrics) - 1)
            if logy:
                ax.set_yscale('log')
        elif i in [2, 3] and final_losses:
            ax.plot(np.arange(len(metrics) - sel_idxs, len(metrics)),
                    metrics[-sel_idxs:, i],
                    color='#1f77b4' if i == 2 else '#ff7f0e',
                    label='valid' if i == 3 else 'train')
            ax.set_title('final losses')
            ax.set_xlim(len(metrics) - sel_idxs, len(metrics) - 1)
            if logy:
                ax.set_yscale('log')
            # ax.set_xticks(np.arange(len(metrics) - sel_idxs, len(metrics)))
        else:
            ax.plot(metrics[:, i], color='#1f77b4' if i == 0 else '#ff7f0e',
                    label='valid' if i > 0 else 'train')
            ax.set_title(name if i >= 2 * (1 + final_losses) else 'losses')
            ax.set_xlim(0, len(metrics) - 1)
            if logy and any(f in name for f in ['mae', 'rmse']):
                ax.set_yscale('log')
            if any(f in name for f in ['r2']):
                ax.set_ylim(top=1)
        ax.legend(loc='best')
        ax.grid(color='gainsboro', linewidth=.5)
    return fig, axs


@patch
@delegates(subplots)
def plot_metrics(self: Learner, **kwargs):
    return self.recorder.plot_metrics(**kwargs)


# %%

# class WeightedPerSampleLoss(Callback):
#     order = 65

#     r"""Loss wrapper than applies a weight per sample during training

#     Weights are not applied to the validation loss.

#     Args:
#         instance_weights:   weights that will be applied. Weights will be
#                             normalized to 1.
#                             You can pass weights for the entire dataset or
#                             just for the training set.
#     """

#     def __init__(self, instance_weights):
#         store_attr()

#     def before_fit(self):
#         self.old_loss = self.learn.loss_func
#         self.reduction = getattr(self.learn.loss_func, 'reduction', None)
#         self.learn.loss_func = _PerInstanceLoss(crit=self.learn.loss_func)
#         if len(self.instance_weights) == len(self.learn.dls.train.dataset):
#             self.instance_weights = torch.cat([
#                 self.instance_weights,
#                 torch.zeros(len(self.learn.dls.valid.dataset))])
#         assert len(self.instance_weights) == \
#             len(self.learn.dls.train.dataset) + \
#             len(self.learn.dls.valid.dataset)
#         self.instance_weights = \
#             self.instance_weights / torch.sum(self.instance_weights) * \
#             len(self.instance_weights)
#         self.instance_weights = torch.as_tensor(
#             self.instance_weights, device=self.learn.dls.device)

#     def before_batch(self):
#         self.learn.loss_func.training = self.training
#         if self.training:
#             input_idxs = self.learn.dls.train.input_idxs
#             self.learn.loss_func.weights = self.instance_weights[input_idxs]

#     def after_fit(self):
#         self.learn.loss_func = self.old_loss
#         if self.reduction is not None:
#             self.learn.loss_func.reduction = self.reduction


# class _PerInstanceLoss(Module):
#     def __init__(self, crit):
#         self.crit = crit
#         self.crit.reduction = 'none'
#         self.weights = None
#         self.training = False

#     def forward(self, input, target):
#         if not self.training:
#             return self.crit(input, target).mean()
#         else:
#             return ((self.crit(input, target) * self.weights)).mean()
