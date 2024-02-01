#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %%
# Imports

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas.api.types import union_categoricals
from scipy import stats
from typing import Any

# %%
# Get the dictionary containing file and path to file


def listdir_path(path: str, do_sort: bool = False):
    import os
    files_path = os.listdir(path)
    if do_sort:
        files_path = list(sorted(files_path))
    return {f: os.path.join(path, f) for f in files_path}


# %%
# Reorder list of strings


def reorder_list(
        lst: list[str], first_elems: list[str] = [],
        last_elems: list[str] = [], drop_elems: list[str] = []) -> list[str]:
    lst = [
        elem for elem in lst
        if elem not in first_elems + drop_elems + last_elems]
    new_order = first_elems + lst + last_elems
    return new_order


# %%
# Concatenation of dataframes with categoricals kept intact


# source: https://stackoverflow.com/a/57809778
def concatenate(dfs: list[pd.DataFrame]):
    """Concatenate while preserving categorical columns.

    NB: We change the categories in-place for the input dataframes"""
    # Iterate on categorical columns common to all dfs
    for col in set.intersection(*[  # type: ignore
            set(df.select_dtypes(include='category').columns)
            for df in dfs]):
        # Generate the union category across dfs for this column
        uc = union_categoricals([df[col] for df in dfs])
        # Change to union category for all dataframes
        for df in dfs:
            df[col] = pd.Categorical(df[col].values, categories=uc.categories)
    return pd.concat(dfs)


# %%


def round_to(n: int, precision: int) -> int:
    correction = 0.5 if n >= 0 else -0.5
    return int((n / precision) + correction) * precision


# %%


def get_distr_density(x: str, y: str, data: pd.DataFrame) -> dict[str, Any]:
    values_kde: npt.NDArray[np.float64] = data[[x, y]].values

    kernel: npt.NDArray[np.float64] = stats.gaussian_kde(
        values_kde.T)(values_kde.T)

    kernel_range = [kernel.min(), kernel.max()]
    norm_kernel = plt.Normalize(*kernel_range)
    sm_kernel = plt.cm.ScalarMappable(cmap='viridis', norm=norm_kernel)
    sm_kernel.set_array([])
    return dict(c=kernel, cmap=sm_kernel.cmap)
