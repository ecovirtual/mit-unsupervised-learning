"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    n = X.shape[0]
    K = mixture.mu.shape[0]
    p_x = np.zeros((n, K))
    cu_one_matrix = X != 0
    dims = np.sum(cu_one_matrix, axis=1)

    for i in range(n):
        for k in range(K):
            filter = np.where(X[i, :] != 0)
            p_x[i, k] = np.log(mixture.p[k] + 1e-16) + (dims[i] / 2) * np.log((1 / ((2 * np.pi * mixture.var[k])))) - ((np.inner(
                (X[i, :][filter] - mixture.mu[k, :][filter]), (X[i, :][filter] - mixture.mu[k, :][filter]))) / (2 * mixture.var[k]))

    min_log_sum = logsumexp(p_x, axis=1, keepdims=True)

    post = np.exp(p_x - min_log_sum)
    LL = np.sum(min_log_sum)

    return post, LL


def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    K = mixture.mu.shape[0]
    indicator = X != 0
    mu = mixture.mu
    n_k = np.sum(post, axis=0)
    p = n_k / n

    for k in range(K):
        for col in range(d):
            if np.dot(post[:, k], indicator[:, col]) >= 1:
                mu[k, col] = np.dot(np.multiply(
                    post[:, k], indicator[:, col]), X[:, col]) / np.dot(post[:, k], indicator[:, col])
            # else:
            #     mu[k, col] = mu[k, col]

    normalizer = np.sum(
        post * np.sum(indicator, axis=1, keepdims=True), axis=0)
    temp = np.zeros((n, K))

    for i in range(n):
        for k in range(K):
            filter = np.where(X[i, :] != 0)
            x_cu = X[i, :][filter]
            mu_cu = mu[k, :][filter]
            temp[i, k] = np.dot((x_cu - mu_cu), (x_cu - mu_cu)) * post[i, k]

    sum = temp.sum(axis=0)
    var = np.maximum((sum / normalizer), min_variance)

    return GaussianMixture(mu, var, p)


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


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    raise NotImplementedError
