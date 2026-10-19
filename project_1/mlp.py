"""Small MLP in numpy (relu, dropout, adam), used together with the logistic regression."""

import numpy as np

from cross_validation import best_threshold, f1_score
from implementations import sigmoid


def init_params(dims, rng):
    """he initialization for the weights, zeros for the biases."""
    W = [(rng.randn(a, b) * np.sqrt(2 / a)).astype(np.float32) for a, b in zip(dims[:-1], dims[1:])]
    b = [np.zeros(d, dtype=np.float32) for d in dims[1:]]
    return W, b


def forward(X, W, b, drop=0.0, rng=None):
    """returns the activations of every layer, the dropout masks and the scores (logits)."""
    acts, masks = [X], []
    for i in range(len(W) - 1):
        a = np.maximum(acts[-1].dot(W[i]) + b[i], 0)
        if drop > 0:
            m = (rng.rand(*a.shape) > drop).astype(np.float32) / (1 - drop)
            a = a * m
            masks.append(m)
        acts.append(a)
    return acts, masks, (acts[-1].dot(W[-1]) + b[-1])[:, 0]


def train_mlp(X, y, X_val, y_val, seed, hidden=(128, 64), epochs=8, lr=1e-3,
              wd=1e-4, drop=0.3, batch_size=512):
    """Train with adam and minibatches, keep the epoch with the best F1 on the validation set.

    returns the weights and biases of that epoch.
    """
    rng = np.random.RandomState(seed)
    W, b = init_params([X.shape[1], *hidden, 1], rng)
    params = W + b
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    best_f1, best = -1, None
    step = 0
    for epoch in range(epochs):
        perm = rng.permutation(len(X))
        for s in range(0, len(perm), batch_size):
            idx = perm[s : s + batch_size]
            acts, masks, z = forward(X[idx], W, b, drop, rng)
            g = ((sigmoid(z) - y[idx]) / len(idx))[:, None].astype(np.float32)
            grads = [None] * len(params)
            for i in range(len(W) - 1, -1, -1):
                grads[i] = acts[i].T.dot(g) + wd * W[i]
                grads[len(W) + i] = g.sum(axis=0)
                if i > 0:
                    g = g.dot(W[i].T) * (acts[i] > 0) * masks[i - 1]
            step += 1
            for j in range(len(params)):
                m[j] = 0.9 * m[j] + 0.1 * grads[j]
                v[j] = 0.999 * v[j] + 0.001 * grads[j] ** 2
                m_hat = m[j] / (1 - 0.9**step)
                v_hat = v[j] / (1 - 0.999**step)
                params[j] -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        scores = forward(X_val, W, b)[2]
        f1 = f1_score(y_val, scores > best_threshold(y_val, scores))
        print("  mlp seed %d epoch %d val f1 %.4f" % (seed, epoch, f1))
        if f1 > best_f1:
            best_f1 = f1
            best = ([w.copy() for w in W], [c.copy() for c in b])
    return best


def predict_scores(X, W, b):
    return forward(X, W, b)[2]
