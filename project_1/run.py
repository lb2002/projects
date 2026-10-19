"""Train the final model on the full training set and write submission.csv."""

import numpy as np

from helpers import create_csv_submission
from preprocessing import load_data, load_names, clean_codes, preprocess
from cross_validation import reg_logistic_newton, best_threshold, f1_score

np.random.seed(1)

x_train, x_test, y_train, train_ids, test_ids = load_data("data")
names = load_names("data")
y = (y_train == 1) * 1.0

x_train = clean_codes(x_train, names)
x_test = clean_codes(x_test, names)
# best setting from cross_validation.py: keep all columns, lambda 1e-5
tx_train, tx_test = preprocess(x_train, x_test, names, max_missing=1.0)
print("features:", tx_train.shape[1])

w = reg_logistic_newton(y, tx_train, 1e-5)

# same as in the cross validation, the threshold is chosen on the training scores
scores = tx_train.dot(w)
thr = best_threshold(y, scores)
print("threshold %.4f, train f1 %.4f" % (thr, f1_score(y, scores > thr)))

y_pred = np.where(tx_test.dot(w) > thr, 1, -1)
print("predicted positive rate %.4f" % np.mean(y_pred == 1))
create_csv_submission(test_ids, y_pred, "submission.csv")
