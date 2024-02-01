#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %%
# Imports

import pandas as pd

from utils.args import getparam

import os

from utils.generic import listdir_path
# %%
# Paths

parent_dir = os.path.dirname(os.getcwd())

dataset_path = os.path.join(parent_dir, 'dataset')

merged_path = os.path.join(dataset_path, 'merged')

# %%
# Which region and data to load

which_region = getparam('region', 'geysers')

# which_data = getparam('data', 'train')
# which_data = getparam('which_data', 'test')
which_data = getparam('which_data', 'test2')

if which_region == 'geysers':
    if which_data == 'train':
        data_geysers_file = 'Data_Geysers_2013_BSSA_mod.txt'
    elif which_data == 'test':
        data_geysers_file = 'training_data.txt'
    elif which_data == 'test2':
        data_geysers_file = 'trainee_data_new.txt'
    else:
        raise ValueError('Wrong data.')
else:
    raise ValueError('Wrong region.')

out_file = f'orig_{which_region}_{which_data}.pkl.bz2'

# %%
# Load of the data

field_values_file = 'field_values_mod.txt'

files_dict = listdir_path(os.path.join(
    dataset_path, 'original'), do_sort=True)

with open(files_dict[field_values_file], 'r') as f:
    field_values = f.readlines()
field_values = [s.strip() for s in field_values if s != '\n']
field_values_spl = [s.split() for s in field_values]

cols_ids_s, cols_names = tuple(
    field_values_spl[i] + field_values_spl[i + 2] for i in range(
        len(field_values_spl) // 2))

cols_ids = [int(col) - 1 for col in cols_ids_s]

col_map = dict(zip(cols_ids, cols_names))

data_geysers = pd.read_csv(
    files_dict[data_geysers_file], sep=' ', header=None)

data_geysers = data_geysers.rename(columns=col_map)

# %%

columns_description = {
    'YY': 'year',
    'MM': 'month',
    'DD': 'day',
    'HR': 'hour',
    'MIN': 'minute',
    'SEC': 'second',
    'EVLA': 'event_lat',
    'EVLO': 'event_lon',
    'STLA': 'station_lat',
    'STLO': 'station_lon',
    'EPDST': 'epicentre_dist',
    'DEPT': 'event_depth',
    'STNM': 'station_name',
    'MAG': 'magnitude',
    'PGV-Z': 'PGV-Z',
    'PGV-N': 'PGV-N',
    'PGV-E': 'PGV-E',
    'PGA-Z': 'PGA-Z',
    'PGA-N': 'PGA-N',
    'PGA-E': 'PGA-E',
    'EID': 'event_ID',
    'SA-Z-0.2': 'SA-Z_0.2s',
    'SA-Z-0.5': 'SA-Z_0.5s',
    'SA-Z-1.0': 'SA-Z_1.0s',
    'SA-N-0.2': 'SA-N_0.2s',
    'SA-N-0.5': 'SA-N_0.5s',
    'SA-N-1.0': 'SA-N_1.0s',
    'SA-E-0.2': 'SA-E_0.2s',
    'SA-E-0.5': 'SA-E_0.5s',
    'SA-E-1.0': 'SA-E_1.0s',
}

# %%
# Get the date

orig_datetime_cols = list(data_geysers.columns[:6])
cur_date_cols_map = {
    columns_description[k]: k for k in data_geysers.columns[:6]}


def get_datetime(df, date_cols_map):
    res = pd.Series('', index=df.index, dtype='O')
    if 'year' in date_cols_map.keys():
        res += df[date_cols_map['year']].apply(lambda v: f'{v:04}') + '-'
        res += df[date_cols_map['month']].apply(lambda v: f'{v:02}') + '-'
        res += df[date_cols_map['day']].apply(lambda v: f'{v:02}')

    if 'hour' in date_cols_map.keys():
        if not res.isna().any():
            res += ' '
        res += df[date_cols_map['hour']].apply(lambda v: f'{v:02}') + ':'
        res += df[date_cols_map['minute']].apply(lambda v: f'{v:02}') + ':'
        res += df[date_cols_map['second']].apply(lambda v: f'{v:02}')

    return pd.to_datetime(res, format='%Y-%m-%d %H:%M:%S')


# %%
# Datetime generation
data_geysers.insert(0, 'datetime', get_datetime(
    data_geysers, date_cols_map=cur_date_cols_map))
data_geysers = data_geysers.drop(columns=cur_date_cols_map.values())

if which_data == 'train':
    assert (data_geysers == data_geysers.sort_values(
        ['datetime', 'EID', 'STNM'])).all().all()
    assert data_geysers.datetime.is_monotonic
else:
    data_geysers = data_geysers.sort_values(
        ['datetime', 'EID', 'STNM'])

data_geysers = data_geysers.set_index('datetime')

# %%
# Save of the original data to disk
os.makedirs(merged_path, exist_ok=True)

data_geysers.to_pickle(os.path.join(merged_path, out_file), protocol=-1)

# %%
