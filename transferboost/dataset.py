import pkgutil
import io
import pandas as pd


def load_data(return_X_y=False, as_frame=False):
    """Loads a simulated data set with two targets, y_1 and y_2.

    y_1 and y_2 are generated
    Args:
        return_X_y:  (bool) If True, returns ``(data, target)`` instead of a dict object.
        as_frame: (bool) give the pandas dataframe instead of X, y matrices (default=False).
    Returns: (pd.DataFrame, dict or tuple) features and target, with as follows:
        - if as_frame is True: returns pd.DataFrame with y as a target
        - return_X_y is True: returns a tuple: (X,y)
        - is both are false (default setting): returns a dictionary where the key `data` contains the features,
        and the key `target` is the target
    """
    file = pkgutil.get_data("transferboost", "data/data.zip")
    df = pd.read_csv(io.BytesIO(file), compression="zip")

    if as_frame:
        return df

    X, y1, y2 = (
        df[[col for col in df.columns if col.startswith("f_")]],
        df["y_1"],
        df["y_2"],
    )
    if return_X_y:
        return X, y1, y2

    return {"data": X, "target_1": y1, "target_2": y2}
