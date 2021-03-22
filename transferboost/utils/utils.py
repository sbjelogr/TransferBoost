import numpy as np
import pandas as pd
import numbers
from .exceptions import DimensionalityError


def check_1d(x):
    """
    Checks whether or not a list, numpy array, pandas dataframe, pandas series are one-dimensional.

    Returns True when check is ok, otherwise throws a `DimensionalityError`.

    Args:
        x: list, numpy array, pandas dataframe, pandas series
    Returns: True or throws `DimensionalityError`
    """
    if isinstance(x, list):
        if any([isinstance(el, list) for el in x]):
            raise DimensionalityError("The input is not 1D")
        else:
            return True
    if isinstance(x, np.ndarray):
        if x.ndim == 1 and all([isinstance(el, numbers.Number) for el in x]):
            return True
        else:
            raise DimensionalityError("The input is not 1D")
    if isinstance(x, pd.core.frame.DataFrame):
        if len(x.columns) == 1 and pd.api.types.is_numeric_dtype(x[x.columns[0]]):
            return True
        else:
            raise DimensionalityError("The input is not 1D")
    if isinstance(x, pd.core.series.Series):
        if x.ndim == 1 and pd.api.types.is_numeric_dtype(x):
            return True
        else:
            raise DimensionalityError("The input is not 1D")


def assure_numpy_array(x, assure_1d=False):
    """
    Returns x as numpy array. X can be a list, numpy array, pandas dataframe, pandas series.

    Args:
        x: list, numpy array, pandas dataframe, pandas series
        assure_1d: whether or not to assure that the input x is one-dimensional
    Returns: numpy array
    """
    if assure_1d:
        _ = check_1d(x)
    if isinstance(x, list):
        return np.array(x)
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, pd.core.frame.DataFrame):
        if len(x.columns) == 1:
            return x.values.flatten()
        else:
            return x.values
    if isinstance(x, pd.core.series.Series):
        return x.values


def assure_pandas_df(x, column_names=None):
    """
    Returns x as pandas DataFrame. X can be a list, list of lists, numpy array, pandas DataFrame or pandas Series.

    Args:
        x (list, numpy array, pandas DataFrame, pandas Series): array to be tested

    Returns:
        pandas DataFrame
    """
    if isinstance(x, pd.DataFrame):
        # Check if column_names are passed correctly
        if column_names is not None:
            x.columns = column_names
        return x
    elif any([isinstance(x, np.ndarray), isinstance(x, pd.core.series.Series), isinstance(x, list)]):
        return pd.DataFrame(x, columns=column_names)
    else:
        raise TypeError("Please supply a list, numpy array, pandas Series or pandas DataFrame")


def check_numeric_dtypes(x):
    """
    Checks if all entries in an array are of a data type that can be interpreted as numeric (int, float or bool).

    Args:
        x (np.ndarray or pd.Series, list): array to be checked

    Returns:
        x: unchanged input array

    Raises:
        TypeError: if not all elements are of numeric dtypes
    """
    x = assure_numpy_array(x)
    allowed_types = [bool, int, float]

    for element in np.nditer(x):
        if type(element.item()) not in allowed_types:
            raise TypeError("Please supply an array with only floats, ints or booleans")
    return x
