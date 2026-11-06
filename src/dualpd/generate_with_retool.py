"""Compatibility alias for :mod:`data.retool.harness`."""

import sys

from dualpd.data.retool import harness as _implementation


sys.modules[__name__] = _implementation
