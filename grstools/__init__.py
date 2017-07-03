
# This file is part of grstools.
#
# The MIT License (MIT)
#
# Copyright (c) 2017 Marc-Andre Legault
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


import logging

try:
    from .version import grstools_version as __version__
except ImportError:
    __version__ = None


__author__ = "Marc-Andre Legault"
__copyright__ = "Copyright 2017 Marc-Andre Legault"
__credits__ = ["Marc-Andre Legault", "Louis-Philippe Lemieux Perreault"]
__license__ = "MIT"
__maintainer__ = "Marc-Andre Legault"
__email__ = "legaultmarc@gmail.com"
__status__ = "Development"


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)


formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s:%(name)s:%(message)s",
    "%H:%M:%S"
)
ch.setFormatter(formatter)
logger.addHandler(ch)
