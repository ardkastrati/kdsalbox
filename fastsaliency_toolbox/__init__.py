#!/usr/bin/env python
"""Top-level module for saliency-toolbox"""

from .version import version as __version__

from .backend.interface import Interface
from .backend.config import Config
from .backend.utils import save_image