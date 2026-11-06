"""Compatibility alias for :mod:`data.retool.tools`."""

import sys

from dualpd.data.retool import tools as _implementation


sys.modules[__name__] = _implementation
