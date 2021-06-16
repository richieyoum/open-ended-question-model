import pickle
import numpy as np
import pandas as pd
import gzip
import os
from typing import Dict


def load_amazon_data(data_dir: str) -> pd.DataFrame:
    """Read in amazon data from data directory

    Args:
        data_dir: data directory

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
        fp: filepath to glove file

    Returns:
        Dict: loaded glove embedding
    """

    glove_w2v = {}
    with open(fp, 'r', encoding='utf-8') as f:
        for line in f:
            splitlines = line.split()
            word = splitlines[0]
            vec = np.array([float(value) for value in splitlines[1:]])
            glove_w2v[word] = vec
    return glove_w2v


def get_embedding_matrix(glove: Dict, word_index: Dict, emb_dim: int) -> Dict:
    """ Get embedding matrix from pretrained glove dict

    Args:
        glove: glove w2v dict. Can get from load_glove function
        word_index: word index from tokenizer
        emb_dim: embedding dimension

    Returns:
        embedding matrix from glove dict
    """
    # populate embedding matrix using glove w2v
    embedding_matrix = np.zeros((len(word_index) + 1, emb_dim))
    for word, i in word_index.items():
        embedding_vector = glove.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def use_pkl(fp: str, mode: str, obj=None):
    """save objects as pickle or read pickle objects

    Args:
        fp (str): [description]
        mode (str): standard file mode; option of 'rb', 'r' for read and 'wb', 'w for write
        obj ([type], optional): Object to pickle. Only relevant when one of the write modes is selected. Defaults to None.
    """
    if obj is None:
        with open(fp, mode) as f:
            return pickle.load(f)
    else:
        with open(fp, mode) as f:
            pickle.dump(obj, f)
