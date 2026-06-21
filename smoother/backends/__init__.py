#
# Copyright (C) 2024-2026 smoother.py.
#
# smoother.py is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Backends providing smooth methods for different data types."""

from importlib import import_module
from pkgutil import iter_modules


def init_backends():
    """Register backends available."""
    for module in iter_modules(__path__):
        try:
            # try to import the model. if dependencies are available,
            # import must work fine
            import_module(f"{__name__}.{module.name}")

        except ModuleNotFoundError as error:
            # a backend with a missing optional dependency is skipped, but a
            # failure inside smoother itself (e.g. missing compiled module)
            # must surface
            if error.name and error.name.startswith("smoother"):
                raise
