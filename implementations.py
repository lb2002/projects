"""ML methods for project 1."""

import numpy as np


def compute_mse(y, tx, w):
    """mse with the 1/2 factor, as in the lecture notes."""
    e = y - tx.dot(w)
    return e.dot(e) / (2 * len(e))


def compute_gradient(y, tx, w):
    """gradient of the mse loss at w."""
    e = y - tx.dot(w)
    return -tx.T.dot(e) / len(e)


def sigmoid(t):
    """apply sigmoid function on t."""
    return 1 / (1 + np.exp(-t))


def compute_logistic_loss(y, tx, w):
    """negative log likelihood averaged over the samples, y in {0,1}."""
    t = tx.dot(w)
    # log(1 + exp(t)) written so it does not overflow for large t
    return np.mean(np.logaddexp(0, t) - y * t)


def compute_logistic_gradient(y, tx, w):
    """gradient of the logistic loss at w."""
    return tx.T.dot(sigmoid(tx.dot(w)) - y) / len(y)


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm for linear regression with mse loss.

    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: shape=(D, ), the last iterate
        loss: the mse loss of w (scalar)
    """
    w = initial_w
    for n_iter in range(max_iters):
        w = w - gamma * compute_gradient(y, tx, w)
    loss = compute_mse(y, tx, w)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD) with batch size 1.

    Same arguments as mean_squared_error_gd, max_iters is the number of SGD steps.
    Returns the last w and its mse loss on the full data.
    """
    w = initial_w
    for n_iter in range(max_iters):
        n = np.random.randint(len(y))
        # same formula as the full gradient, just on one sample
        w = w - gamma * compute_gradient(y[n : n + 1], tx[n : n + 1], w)
    loss = compute_mse(y, tx, w)
    return w, loss


def least_squares(y, tx):
    """Calculate the least squares solution.
       returns optimal weights and mse.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.
    """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    return w, compute_mse(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar, without the penalty term.
    """
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    return w, compute_mse(y, tx, w)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression with gradient descent, labels y in {0,1}.

    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        initial_w: shape=(D, )
        max_iters: number of GD steps
        gamma: stepsize

    Returns:
        w: shape=(D, ), the last iterate
        loss: negative log likelihood of w (scalar)
    """
    w = initial_w
    for n_iter in range(max_iters):
        w = w - gamma * compute_logistic_gradient(y, tx, w)
    loss = compute_logistic_loss(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Penalized logistic regression with gradient descent.

    The penalty is lambda_ * ||w||^2, so 2 * lambda_ * w is added to the gradient.
    Same arguments as logistic_regression plus lambda_.
    The returned loss does not include the penalty.
    """
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_logistic_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * grad
    loss = compute_logistic_loss(y, tx, w)
    return w, loss
