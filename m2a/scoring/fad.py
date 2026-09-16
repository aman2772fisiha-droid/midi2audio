"""Fréchet Audio Distance (FAD) distributional scoring against reference corpora."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import numpy as np
from scipy import linalg


def compute_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Calculate Fréchet distance between two multivariate Gaussian distributions."""
    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            covmean = covmean.real
        else:
            covmean = covmean.real

    tr_covmean = np.trace(covmean)
    fad = float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)
    return max(0.0, fad)


def compute_distribution_statistics(embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean vector and covariance matrix from collection of embedding vectors."""
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embedding matrix (N, D), received shape {embeddings.shape}")
    mu = np.mean(embeddings, axis=0)
    sigma = np.cov(embeddings, rowvar=False)
    sigma += np.eye(sigma.shape[0]) * 1e-6
    return mu, sigma


def calculate_fad_from_embeddings(ref_embeddings: np.ndarray, eval_embeddings: np.ndarray) -> float:
    """Calculate FAD between reference corpus and generated candidate window embeddings."""
    if len(ref_embeddings) < 2 or len(eval_embeddings) < 2:
        raise ValueError("At least 2 samples required per distribution to estimate covariance.")

    mu_ref, sigma_ref = compute_distribution_statistics(ref_embeddings)
    mu_eval, sigma_eval = compute_distribution_statistics(eval_embeddings)

    return compute_frechet_distance(mu_ref, sigma_ref, mu_eval, sigma_eval)