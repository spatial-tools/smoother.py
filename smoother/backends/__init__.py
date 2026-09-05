#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <https://www.gnu.org/licenses/>.
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
