"""Cleaning and feature building for the BRFSS data."""

import os
import numpy as np

from helpers import load_csv_data

# ids, dates, sampling weights, and raw questions where a calculated version exists
# (e.g. ALCDAY5 mixes "days per week" and "days per month" in one code, DROCDY3_ is the clean one)
DROP = [
    "FMONTH", "IDATE", "IMONTH", "IDAY", "IYEAR", "SEQNO", "_PSU", "_STSTR",
    "_STRWT", "_RAWRAKE", "_WT2RAKE", "_CLLCPWT", "_DUALCOR", "_LLCPWT",
    "WEIGHT2", "HEIGHT3", "HTIN4", "FLSHTMY2", "HIVTSTD3",
    "ALCDAY5", "FRUITJU1", "FRUIT1", "FVBEANS", "FVGREEN", "FVORANG", "VEGETAB1",
    "EXEROFT1", "EXEROFT2", "EXERHMM1", "EXERHMM2", "STRENGTH",
    "BLDSUGAR", "FEETCHK2", "LONGWTCH",
]  # fmt: skip

# missing codes that don't follow the 7/9 pattern
SPECIAL = {
    "_AGEG5YR": [14],
    "_AGE65YR": [3],
    "DROCDY3_": [900],
    "_DRNKWEK": [99900],
    "MAXVO2_": [999],
    "FC60_": [999],
}

# categorical even though they have many values
FORCE_CAT = ["_STATE", "EXRACT11", "EXRACT21"]


def load_data(data_path="data"):
    """load_csv_data takes ~2 min, so keep a .npz copy next to the csv files."""
    cache = os.path.join(data_path, "cache.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["x_train"], d["x_test"], d["y_train"], d["train_ids"], d["test_ids"]
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(data_path)
    np.savez(
        cache,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        train_ids=train_ids,
        test_ids=test_ids,
    )
    return x_train, x_test, y_train, train_ids, test_ids


def load_names(data_path="data"):
    """column names of x, without the Id column."""
    with open(os.path.join(data_path, "x_train.csv")) as f:
        return f.readline().strip().split(",")[1:]


def clean_codes(x, names):
    """BRFSS "don't know" / "refused" codes -> nan, "none" -> 0.

    the codes depend on the width of the field (7/9, 77/99, 777/999), so look at the max of the column
    """
    x = x.copy()
    for j, name in enumerate(names):
        c = x[:, j]
        if name in SPECIAL:
            c[np.isin(c, SPECIAL[name])] = np.nan
        if name.endswith("_"):
            # calculated continuous variables, 99 can be a real value there
            continue
        m = np.nanmax(c)
        if m == 9:
            c[c == 9] = np.nan
            # calculated variables only use 9. for EMPLOY1 7 means retired
            if not name.startswith("_") and name != "EMPLOY1":
                c[c == 7] = np.nan
        elif m in (99, 999, 999999):
            ones = (m + 1) // 9  # 11, 111, ...
            c[(c == 7 * ones) | (c == 9 * ones)] = np.nan
            c[c == 8 * ones] = 0
            if m == 999:
                c[c == 555] = 0  # "never" in the x times per day/week/month questions
    return x


def fit_preprocessing(x, names, max_missing=0.5, max_unique=10, onehot=True):
    """Decide per column what to do with it, on the cleaned training data.

    returns a list of (j, "cat", levels) or (j, "cont", (median, mean, std))
    """
    params = []
    for j, name in enumerate(names):
        c = x[:, j]
        nan = np.isnan(c)
        if name in DROP or nan.mean() > max_missing:
            continue
        levels = np.unique(c[~nan])
        if len(levels) < 2:
            continue
        if onehot and (len(levels) <= max_unique or name in FORCE_CAT):
            params.append((j, "cat", levels))
        else:
            median = np.median(c[~nan])
            c = np.where(nan, median, c)
            std = c.std()
            if std > 0:
                params.append((j, "cont", (median, c.mean(), std)))
    return params


def apply_preprocessing(x, params, degree=1, nan_indicator=False):
    """Build the feature matrix tx (with a leading column of ones) from cleaned x."""
    cols = [np.ones(len(x))]
    for j, kind, info in params:
        c = x[:, j]
        nan = np.isnan(c)
        if kind == "cat":
            # first level is the reference, otherwise the columns sum to the intercept
            for v in info[1:]:
                cols.append((c == v).astype(float))
            cols.append(nan.astype(float))
        else:
            median, mean, std = info
            c = (np.where(nan, median, c) - mean) / std
            c = np.clip(c, -5, 5)  # a few columns have extreme outliers
            for deg in range(1, degree + 1):
                cols.append(c**deg)
            if nan_indicator:
                cols.append(nan.astype(float))
    return np.column_stack(cols)


def drop_useless(tx_train, tx_other):
    """remove constant and duplicated columns"""
    keep = tx_train.std(axis=0) > 0
    keep[0] = True
    # duplicates: comparing all pairs of columns is too slow, so compare one random projection
    r = np.random.RandomState(0).randn(len(tx_train))
    _, first = np.unique(np.round(r.dot(tx_train), 6), return_index=True)
    uniq = np.zeros(tx_train.shape[1], dtype=bool)
    uniq[first] = True
    keep = keep & uniq
    return tx_train[:, keep], tx_other[:, keep]


def preprocess(x_train, x_other, names, **kwargs):
    """fit on x_train, apply to both (both already cleaned)."""
    fit_keys = ["max_missing", "max_unique", "onehot"]
    fit_kwargs = {k: v for k, v in kwargs.items() if k in fit_keys}
    apply_kwargs = {k: v for k, v in kwargs.items() if k not in fit_keys}
    params = fit_preprocessing(x_train, names, **fit_kwargs)
    tx_train = apply_preprocessing(x_train, params, **apply_kwargs)
    tx_other = apply_preprocessing(x_other, params, **apply_kwargs)
    return drop_useless(tx_train, tx_other)
