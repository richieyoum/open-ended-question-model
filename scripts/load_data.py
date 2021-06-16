import numpy as np
import pandas as pd
import gzip
import os
from typing import Dict


def load_amazon_data(data_dir: str) -> pd.DataFrame:
    """Read in amazon data from data directory

    Args:
        data_dir (str): data directory

    Returns:
        pd.DataFrame: amazon data
  """

    def parse(path):
        g = gzip.open(path, 'rb')
        for l in g:
            yield eval(l)

    def getDF(path):
        i = 0
        df = {}
        for d in parse(path):
            df[i] = d
            i += 1
        return pd.DataFrame.from_dict(df, orient='index')

    dfs = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            dfs.append(getDF(os.path.join(root, f)))

    df_all = pd.concat(dfs)
    return df_all


def load_glove(fp: str) -> Dict:
    """Load glove embedding from filepath

  Args:
      fp (str): filepath to glove file

  Returns:
      Dict: loaded glove embedding
  """

    glove_w2v = {}
    with open(fp, 'r', encoding='utf-8') as f:
        for line in f:
            splitLines = line.split()
            word = splitLines[0]
            vec = np.array([float(value) for value in splitLines[1:]])
            glove_w2v[word] = vec
    return glove_w2v
