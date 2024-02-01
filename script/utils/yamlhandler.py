#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 11:35:36 2021

@author: modal
"""

# %%

import yaml


class PrettySafeLoader(yaml.SafeLoader):
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))


PrettySafeLoader.add_constructor(
    u'tag:yaml.org,2002:python/tuple',
    PrettySafeLoader.construct_python_tuple)


def safe_load(stream):
    return yaml.load(stream, Loader=PrettySafeLoader)


class PrettySafeDumper(yaml.SafeDumper):
    def represent_python_tuple(dumper, node):
        return dumper.represent_sequence(
            u'tag:yaml.org,2002:python/tuple', node)


PrettySafeDumper.add_representer(
    tuple, PrettySafeDumper.represent_python_tuple)


def safe_dump(data, stream=None, **kwds):
    return yaml.dump(data, stream, Dumper=PrettySafeDumper, **kwds)


class YamlHandler:
    @staticmethod
    def safe_load_file(filepath):
        with open(filepath, 'r') as yaml_file:
            return safe_load(yaml_file)

    @staticmethod
    def safe_dump_file(
            data, filepath, default_flow_style=False, sort_keys=False, **kwds):
        with open(filepath, 'w') as yaml_file:
            return safe_dump(
                data, yaml_file, default_flow_style=default_flow_style,
                sort_keys=sort_keys, **kwds)
