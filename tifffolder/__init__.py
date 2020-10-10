# -*- coding: utf-8 -*-
"""
    tifffolder
    ~~~~~~~~~~
    A package for parsing filenames in a folder of images and accessing
    them lazily using numpy indexing syntax
    :copyright: © 2018, Talley Lambert
    :license: MIT, see LICENSE for more details.
"""

__version__ = '0.1.8'

from .tifffolder import TiffFolder, LLSFolder, MetamorphFolder, build_regex, imshow
from .axesarray import AxesArray
