from __future__  import print_function

import sys

__version__ = '1.0.0'

try:
    import pyartm
except ImportError:
    print(
        'You have to install pyartm to perform the pyartm_experiments',
        file=sys.stderr
    )

try:
    import pyartm_datasets
except ImportError:
    print(
        'You have to install pyartm_datasets to perform the pyartm_experiments',
        file=sys.stderr
    )
