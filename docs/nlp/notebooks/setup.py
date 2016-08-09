from __future__ import division, print_function, absolute_import, unicode_literals
from builtins import str  # noqa

import os
import sys
import gzip
import collections
import string
from collections import Counter, OrderedDict, Mapping
from decimal import Decimal

import matplotlib
import matplotlib.pyplot as plt

# for ipython notebooks
from IPython.display import display, HTML
# import pandas_profiling

from constants import *

from pandas.tools.plotting import scatter_matrix

from gensim.models import TfidfModel, LsiModel
from gensim.corpora import Dictionary


def scatmat(df, category=None, colors='rgyb',
            num_plots=4, num_topics=100, num_columns=4,
            show=False, block=False, data_path=DATA_PATH, save=False, verbose=1):
    """FIXME: empty plots that dont go away, Plot and/save scatter matrix in groups of num_columns topics"""

    if category is None:
        category = list(df.columns)[-1]
    if category in df.columns:
        category = df[category]
    else:
        category = pd.Series(category)

    suffix = '{}x{}'.format(len(df), num_topics)
    save = bool(save)
    for i in range(int(min(num_plots * num_columns, num_topics) / float(num_plots))):
        scatter_matrix(df[df.columns[i * num_columns:(i + 1) * num_columns]],
                       marker='+', c=[colors[int(x) % len(colors)] for x in category.values],
                       figsize=(18, 12))
        if save:
            name = 'scatmat_topics_{}-{}.jpg'.format(i * num_columns, (i + 1) * num_columns) + suffix
            plt.savefig(os.path.join(data_path, name + '.jpg'))
        if show:
            if block:
                plt.show()
            else:
                plt.show(block=False)
