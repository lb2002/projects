"""Train the final model on the training set and write a submission csv.

usage: python run.py [name] [lr]
the file is written to submissions/name.csv, the default model is the blend of
regularized logistic regression and 3 mlps, with "lr" only the logistic regression is used
"""

import os
import sys
import numpy as np

from helpers import create_csv_submission
from preprocessing import load_data, load_names, clean_codes, preprocess
from cross_validation import reg_logistic_newton, best_threshold, f1_score
from mlp import train_mlp, predict_scores

np.random.seed(1)

name = sys.argv[1] if len(sys.argv) > 1 else "submission"
use_mlp = not (len(sys.argv) > 2 and sys.argv[2] == "lr")
out = os.path.join("submissions", name + ".csv")
# dont overwrite old submissions
if os.path.exists(out):
    sys.exit(out + " already exists, pick another name")
os.makedirs("submissions", exist_ok=True)

x_train, x_test, y_train, train_ids, test_ids = load_data("data")
names = load_names("data")
y = (y_train == 1) * 1.0

x_train = clean_codes(x_train, names)
x_test = clean_codes(x_test, names)
# best setting from cross_validation.py: keep all columns, lambda 1e-5
tx_train, tx_test = preprocess(x_train, x_test, names, max_missing=1.0)
print("features:", tx_train.shape[1])

if not use_mlp:
    w = reg_logistic_newton(y, tx_train, 1e-5)

    # same as in the cross validation, the threshold is chosen on the training scores
    scores = tx_train.dot(w)
    thr = best_threshold(y, scores)
    print("threshold %.4f, train f1 %.4f" % (thr, f1_score(y, scores > thr)))
    test_scores = tx_test.dot(w)
else:
    # 1/8 of the training data is kept for early stopping of the mlps and for the threshold
    perm = np.random.RandomState(0).permutation(len(y))
    n_val = len(y) // 8
    val, fit = perm[:n_val], perm[n_val:]

    w = reg_logistic_newton(y[fit], tx_train[fit], 1e-5)
    val_lr, test_lr = tx_train[val].dot(w), tx_test.dot(w)

    X_fit = tx_train[fit].astype(np.float32)
    X_val = tx_train[val].astype(np.float32)
    X_test = tx_test.astype(np.float32)
    val_mlp, test_mlp = [], []
    for seed in range(3):
        W, b = train_mlp(X_fit, y[fit].astype(np.float32), X_val, y[val], seed)
        val_mlp.append(predict_scores(X_val, W, b))
        test_mlp.append(predict_scores(X_test, W, b))
    val_mlp, test_mlp = np.mean(val_mlp, axis=0), np.mean(test_mlp, axis=0)

    # bring both scores to the same scale (mean and std of the validation scores), then average
    val_scores = 0.5 * (val_mlp - val_mlp.mean()) / val_mlp.std()
    val_scores += 0.5 * (val_lr - val_lr.mean()) / val_lr.std()
    test_scores = 0.5 * (test_mlp - val_mlp.mean()) / val_mlp.std()
    test_scores += 0.5 * (test_lr - val_lr.mean()) / val_lr.std()
    # the f1 curve is very flat on the small validation set (12% to 18% positives are all within noise),
    # so use the positive rate of the logistic regression instead: its threshold is tuned on all training
    # samples. The blend then predicts the same number of positives on the test set.
    w_all = reg_logistic_newton(y, tx_train, 1e-5)
    thr_all = best_threshold(y, tx_train.dot(w_all))
    rate = np.mean(tx_test.dot(w_all) > thr_all)
    thr = np.quantile(test_scores, 1 - rate)
    print("positive rate %.4f, threshold %.4f" % (rate, thr))
    print("val f1 of the blend at its best threshold %.4f" % f1_score(y[val], val_scores > best_threshold(y[val], val_scores)))

y_pred = np.where(test_scores > thr, 1, -1)
print("predicted positive rate %.4f" % np.mean(y_pred == 1))
create_csv_submission(test_ids, y_pred, out)
