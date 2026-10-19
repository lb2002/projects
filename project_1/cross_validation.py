"""5-fold cross validation, baselines and the improvement steps. Writes results.csv."""

import csv
import numpy as np

from implementations import *
from preprocessing import load_data, load_names, clean_codes, preprocess

K_FOLD = 5
SEED = 1


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold (same as in lab 4)."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def accuracy(y, pred):
    return np.mean(y == pred)


def f1_score(y, pred):
    """F1 of the positive class, y and pred in {0,1}."""
    tp = np.sum((y == 1) & (pred == 1))
    if tp == 0:
        return 0.0
    precision = tp / np.sum(pred == 1)
    recall = tp / np.sum(y == 1)
    return 2 * precision * recall / (precision + recall)


def best_threshold(y, scores):
    """threshold on the scores that maximizes F1."""
    order = np.argsort(-scores)
    tp = np.cumsum(y[order])
    # f1 when the k highest scores are predicted positive
    f1 = 2 * tp / (np.arange(1, len(y) + 1) + y.sum())
    k = np.argmax(f1)
    return (scores[order[k]] + scores[order[k + 1]]) / 2


def reg_logistic_newton(y, tx, lambda_, max_iters=30, pos_weight=1.0):
    """newton's method for regularized logistic regression, see lab 5.

    pos_weight multiplies the loss of the positive samples.
    """
    N, D = tx.shape
    w = np.zeros(D)
    s = np.where(y == 1, pos_weight, 1.0)
    for n_iter in range(max_iters):
        p = sigmoid(tx.dot(w))
        grad = tx.T.dot(s * (p - y)) / N + 2 * lambda_ * w
        hess = (tx.T * (s * p * (1 - p))).dot(tx) / N + 2 * lambda_ * np.identity(D)
        step = np.linalg.solve(hess, grad)
        w = w - step
        if np.abs(step).max() < 1e-5:
            break
    return w


def undersample(y, tx, ratio, seed=SEED):
    """keep all positives and ratio times as many random negatives."""
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    rng = np.random.RandomState(seed)
    neg = rng.choice(neg, size=int(ratio * len(pos)), replace=False)
    idx = np.concatenate([pos, neg])
    return y[idx], tx[idx]


def select_top_k(y, tx_tr, tx_te, top_k):
    # top_k features by absolute correlation with y, plus the intercept
    yc = y - y.mean()
    txc = tx_tr[:, 1:] - tx_tr[:, 1:].mean(axis=0)
    corr = np.abs(yc.dot(txc)) / (np.sqrt((txc**2).sum(axis=0)) * np.sqrt(yc.dot(yc)))
    keep = np.concatenate([[0], 1 + np.sort(np.argsort(-corr)[:top_k])])
    return tx_tr[:, keep], tx_te[:, keep]


def cross_validate(y, x, names, models, top_k=None, **prep_kwargs):
    """k-fold CV for several models on the same preprocessing.

    models: dict name -> (train, thr0). train(y, tx) returns w, thr0 is the default
    threshold on tx.dot(w) (0.5 for the mse methods, 0 for logistic).
    Returns per model an array (k_fold, 4): acc, f1 with thr0, then acc, f1 with the tuned threshold.
    """
    k_indices = build_k_indices(y, K_FOLD, SEED)
    res = {name: [] for name in models}
    for k in range(K_FOLD):
        te = k_indices[k]
        tr = np.delete(k_indices, k, axis=0).reshape(-1)
        tx_tr, tx_te = preprocess(x[tr], x[te], names, **prep_kwargs)
        if top_k is not None:
            tx_tr, tx_te = select_top_k(y[tr], tx_tr, tx_te, top_k)
        for name, (train, thr0) in models.items():
            w = train(y[tr], tx_tr)
            s_tr, s_te = tx_tr.dot(w), tx_te.dot(w)
            # tuned on the training part only
            thr = best_threshold(y[tr], s_tr)
            res[name].append(
                [
                    accuracy(y[te], s_te > thr0),
                    f1_score(y[te], s_te > thr0),
                    accuracy(y[te], s_te > thr),
                    f1_score(y[te], s_te > thr),
                ]
            )
    print("  D =", tx_tr.shape[1])
    return {name: np.array(r) for name, r in res.items()}


def log_results(step, res, rows):
    for name, r in res.items():
        m, s = r.mean(axis=0), r.std(axis=0)
        print(
            "  %-34s acc %.4f f1 %.4f | tuned thr: acc %.4f f1 %.4f +- %.4f"
            % (name, m[0], m[1], m[2], m[3], s[3])
        )
        rows.append([step, name] + ["%.4f" % v for v in [m[0], m[1], m[2], m[3], s[3]]])


def write_results(rows, name="results.csv"):
    with open(name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["step", "model", "acc", "f1", "acc_tuned_thr", "f1_tuned_thr", "f1_std"]
        )
        writer.writerows(rows)


def w0(tx):
    return np.zeros(tx.shape[1])


if __name__ == "__main__":
    x_train, _, y_train, _, _ = load_data()
    names = load_names()
    y = (y_train == 1) * 1.0
    x_raw = x_train
    x = clean_codes(x_train, names)
    rows = []

    print("positive rate %.4f" % y.mean())

    print("baselines, simple preprocessing")
    baselines = {
        "mse gd (gamma 0.03, 300 it)": (
            lambda y, tx: mean_squared_error_gd(y, tx, w0(tx), 300, 0.03)[0],
            0.5,
        ),
        "mse sgd (gamma 1e-3, 200k it)": (
            lambda y, tx: mean_squared_error_sgd(y, tx, w0(tx), 200000, 1e-3)[0],
            0.5,
        ),
        "least squares": (lambda y, tx: least_squares(y, tx)[0], 0.5),
        "ridge (lambda 1e-4)": (lambda y, tx: ridge_regression(y, tx, 1e-4)[0], 0.5),
        "logistic gd (gamma 0.3, 1000 it)": (
            lambda y, tx: logistic_regression(y, tx, w0(tx), 1000, 0.3)[0],
            0,
        ),
        "reg logistic gd (lambda 1e-4)": (
            lambda y, tx: reg_logistic_regression(y, tx, 1e-4, w0(tx), 1000, 0.3)[0],
            0,
        ),
    }
    log_results("baseline", cross_validate(y, x, names, baselines), rows)
    write_results(rows)

    def newton(lambda_, pos_weight=1.0):
        return (
            lambda y, tx: reg_logistic_newton(y, tx, lambda_, pos_weight=pos_weight),
            0,
        )

    def under(ratio):
        def train(y, tx):
            y_s, tx_s = undersample(y, tx, ratio)
            return reg_logistic_newton(y_s, tx_s, 1e-5)

        return (train, 0)

    # from here on everything is regularized logistic regression trained with newton,
    # one change at a time against "newton (lambda 1e-5)"
    ref = {"newton (lambda 1e-5)": newton(1e-5)}

    def run(step, models=ref, x=x, **kwargs):
        print(step)
        log_results(step, cross_validate(y, x, names, models, **kwargs), rows)
        write_results(rows)

    balance = {
        "newton (lambda 1e-5)": newton(1e-5),
        "class weight 3": newton(1e-5, 3.0),
        "class weight 10.3 (balanced)": newton(1e-5, 10.3),
        "undersample 1:3": under(3),
        "undersample 1:1": under(1),
    }
    run("optimizer and class balance", balance)

    run("no code cleaning", x=x_raw)
    run("no one-hot", onehot=False)
    run("no code cleaning, no one-hot", x=x_raw, onehot=False)

    run("feature selection top 100", top_k=100)
    run("feature selection top 250", top_k=250)

    run("poly degree 2", degree=2)
    run("poly degree 3", degree=3)
    run("nan indicator for continuous", nan_indicator=True)

    run("max_missing 0.3", max_missing=0.3)
    run("max_missing 0.9", max_missing=0.9)
    run("max_missing 1.0 (keep all columns)", max_missing=1.0)

    lambdas = {
        "newton (lambda %g)" % l: newton(l) for l in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    }
    lambdas["ridge (lambda 1e-4)"] = (
        lambda y, tx: ridge_regression(y, tx, 1e-4)[0],
        0.5,
    )
    run("all columns, lambda search", lambdas, max_missing=1.0)
    run("all columns, poly degree 2", max_missing=1.0, degree=2)

    # also tried with all columns (lambda 1e-4), not in the table:
    # degree=3 0.4396, nan_indicator=True 0.4377, top_k=400 0.4337
