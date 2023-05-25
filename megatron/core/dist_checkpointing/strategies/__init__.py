# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

""" Various loading and saving strategies """

try:
    import zarr
    import tensorstore
    from .zarr import _import_trigger
    from .tensorstore import _import_trigger
except ImportError:
    print('Zarr strategies will not be registered because of missing packages')
