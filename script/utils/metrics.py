#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 01:35:42 2021

@author: modal
"""

# %%

import numpy as np
import pandas as pd
import torch
from fastcore.all import *

# %%


def sqrt(r):
    if isinstance(r, torch.Tensor):
        return torch.sqrt(r)
    return np.sqrt(r)


def unique(group):
    if isinstance(group, torch.Tensor):
        return torch.unique(group)
    return np.unique(group)


def zeros_like(v, dtype):
    if isinstance(v, torch.Tensor):
        return torch.zeros(v.size(), dtype=dtype, device=v.device)
    return np.zeros(v.shape, dtype=dtype)


def total_sum_of_squares(r):
    rmean = r.mean()
    return ((r - rmean) ** 2).sum()


def within_group_sum_of_squares(r, group):
    uniq = unique(group)
    group_ss = zeros_like(uniq, dtype=r.dtype)
    for i, val in enumerate(uniq):
        group_ss[i] += total_sum_of_squares(r[group == val])
    return group_ss.sum()


# def between_group_sum_of_squares(r, group):
#     grand_mean = r.mean()
#     mean_group = r.groupby(group).mean()
#     n_group = r.groupby(group).count()

#     return (n_group * ((mean_group - grand_mean) ** 2)).sum()


def total_variance(r):
    return total_sum_of_squares(r) / (len(r) - 1)


def within_group_variance(r, group):
    N = len(r)
    N_group = len(unique(group))
    return within_group_sum_of_squares(r, group) / (N - N_group)


def between_group_variance(r, group):
    totalvar = total_variance(r)
    withinvar = within_group_variance(r, group)
    return totalvar - withinvar


# def between_group_variance(r, group):
#     N_group = len(np.unique(group))
#     return between_group_sum_of_squares(r, group) / (N_group - 1)


def total_sigma(y_true, y_pred):
    return sqrt(total_variance(y_true - y_pred))


def within_sigma(y_true, y_pred, group):
    return sqrt(within_group_variance(y_true - y_pred, group))


def between_sigma(y_true, y_pred, group):
    return sqrt(between_group_variance(y_true - y_pred, group))


class SigmasMetrics:
    def __init__(
            self, dataframe: pd.DataFrame, true_col: str, pred_col: str,
            event_col: str = 'event', station_col: str = 'station',
            resid_col_name: str = None) -> None:
        if resid_col_name is None:
            resid_col_name = f'{true_col}_resid'
        store_attr()
        self._generate_df_resid()

    def _generate_df_resid(self):
        df = self.dataframe[[
            self.event_col, self.station_col, self.true_col,
            self.pred_col]].copy()
        df[self.resid_col_name] = df[self.true_col] - df[self.pred_col]
        self.df_resid = pd.pivot_table(
            df, values=self.resid_col_name,
            index=self.station_col, columns=self.event_col)

    def sigma(self):
        return self.df_resid.stack().std()

    def within_event_var(self):
        rr = self.df_resid
        rjmean = rr.mean(axis=0)

        ddofs = 0
        within_var = 0
        for j in rr.columns:
            for i in rr.index:
                rij = rr.loc[i, j]

                if np.isnan(rij):
                    continue
                within_var += (rij - rjmean.loc[j]) ** 2
                ddofs += 1

        return within_var, ddofs - len(rjmean)

    def within_event_sigma(self):
        within_var, ddofs = self.within_event_var()
        return sqrt(within_var / ddofs)

    def sigma_intraevent(self):
        return float(sqrt(
            self.df_resid.stack().groupby('event').var().mean()))

    def sigma_intraevent2(self):
        var = self.df_resid.stack().groupby('event').var()
        return float(sqrt(
            var.sum() / (var.notna().sum() - 1)))

    def sigma_interevent(self):
        return float(
            sqrt(self.sigma() ** 2 - self.sigma_intraevent() ** 2))

    def sigma_interevent2(self):
        return float(
            sqrt(self.sigma() ** 2 - self.sigma_intraevent2() ** 2))
