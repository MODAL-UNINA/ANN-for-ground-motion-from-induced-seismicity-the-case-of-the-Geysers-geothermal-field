#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %%
# Imports

import sys

# %%
# ipykernel_launcher corresponds to the run done with VSCode

_interactive_mode = 'ipykernel_launcher' in sys.argv[0] or \
    (len(sys.argv) == 1 and sys.argv[0] == '')

if _interactive_mode:
    args = None
else:
    print('Arguments passed:')
    print(sys.argv)

    args = iter(sys.argv[1:])


def is_interactive():
    return _interactive_mode


def enable_interactive():
    global _interactive_mode
    if not _interactive_mode:
        _interactive_mode = not _interactive_mode
    else:
        print('Interactive already enabled.')


def getparam(paramname, defaultval, optional=False):
    if _interactive_mode:
        return defaultval
    dtype = type(defaultval)

    argparam = f'--{paramname}='
    l_argparam = [arg for arg in sys.argv if arg.startswith(argparam)]
    if len(l_argparam) > 1:
        raise ValueError(
            f'Args: argument \'{paramname}\' passed more than once!')

    if len(l_argparam) == 0:
        if optional:
            return defaultval
        raise ValueError(f'Args: missing argument: \'{paramname}\'')

    return dtype(l_argparam[0][len(argparam):])
