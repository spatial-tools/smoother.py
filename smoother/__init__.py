#
# Copyright (C) 2024-2026 smoother.py.
#
# smoother.py is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Smoother module."""

from smoother.backends import init_backends
from smoother.smooth import smooth

#
# Initialize backends
#
init_backends()

#
# Package version
#
__version__ = "0.1.0"

#
# Public API functions
#
__all__ = (
    "__version__",
    "smooth",
)
