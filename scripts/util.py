import pickle
from typing import Any


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
