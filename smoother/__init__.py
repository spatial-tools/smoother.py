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
__version__ = "0.1.3"

#
# Public API functions
#
__all__ = (
    "__version__",
    "smooth",
)
