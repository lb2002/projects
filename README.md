# CS-433 Project 1: predicting MICHD from BRFSS data

Binary classification of coronary heart disease (MICHD) on the BRFSS 2015 survey, using only numpy.

## How to run

Put `x_train.csv`, `y_train.csv` and `x_test.csv` from AIcrowd into `data/`, then

```
python run.py
```

This trains the final model and writes `submission.csv`. The first run parses the csv files (about 2 minutes) and stores them as `data/cache.npz`, later runs take under a minute.

To reproduce the cross validation numbers (takes about 1.5 hours, writes `results.csv`):

```
python cross_validation.py
```

Tests for the 6 required functions, from the course repo:

```
cd grading_tests
pytest --github_link .. .
```

## Files

- `implementations.py` the 6 required methods (GD, SGD, least squares, ridge, logistic, regularized logistic) and their loss/gradient helpers
- `helpers.py` csv loading and submission writing, given by the course
- `preprocessing.py` cleaning of the BRFSS missing codes, one-hot encoding, imputation, standardization
- `cross_validation.py` 5-fold CV, metrics, threshold tuning, baselines and all improvement steps
- `run.py` trains the final model and writes the submission
- `exploration.ipynb` data exploration (shapes, class balance, missing values, special codes)
- `results.csv` output of `cross_validation.py`, one row per experiment
- `grading_tests/`, `latex-example-paper/` copied from the course repo

## Method

Preprocessing is in `preprocessing.py`. The BRFSS codes for "don't know" / "refused" (7/9, 77/99, 777/999) become nan and "none" (88, 888) becomes 0. Ids, dates, sampling weights and raw questions that have a calculated counterpart are dropped. Columns with at most 10 distinct values are one-hot encoded with nan as its own category, the rest gets the median for nan and is standardized (train statistics only) and clipped to [-5, 5].

The final model is regularized logistic regression (lambda 1e-5) on all columns, trained with Newton's method because gradient descent was still far from converged after 1000 steps. Only 8.8% of the labels are positive, so the decision threshold is chosen to maximize F1 on the training data.

Cross validation is 5-fold with seed 1, the threshold is tuned on the training folds.

## Dependencies

Python 3.9+, numpy, matplotlib (notebook only). pytest, gitpython and black for the grading tests.
