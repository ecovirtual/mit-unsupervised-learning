"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n, d = X.shape
    K, _ = mixture.mu.shape
    p_x = np.zeros((n, K))

    for i in range(n):
        for k in range(K):
            p_k = mixture.p[k]
            normalization = (1 / ((2 * np.pi * mixture.var[k])**(d / 2)))
            exp = np.exp(-((np.inner((X[i, :] - mixture.mu[k, :]),
                         (X[i, :] - mixture.mu[k, :]))) / (2 * mixture.var[k])))
            p_x[i, k] = p_k * normalization * exp

    p_theta = p_x.sum(axis=1)

    post = p_x / p_theta.reshape(-1, 1)
    log_likelihood = np.sum(post * np.log((p_x / post)))

    return (post, log_likelihood)


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    raise NotImplementedError


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    raise NotImplementedError
