"""
EFA Core Library
================

Modular analysis framework for Exploratory Factor Analysis and price tier validation.

Modules:
    config  - Global configuration parameters
    data    - Data loading, filtering, standardization
    efa     - Factor analysis functions
    models  - Machine learning models (Random Forest)
    stats   - Statistical tests (ANOVA, KS)
    viz     - Visualization utilities
    output  - Output naming and saving
"""

from . import config
from . import data
from . import efa
from . import models
from . import stats
from . import viz
from . import output

__version__ = '1.0.0'

__all__ = [
    'config',
    'data',
    'efa',
    'models',
    'stats',
    'viz',
    'output',
]
