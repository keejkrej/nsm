"""Bayesian inference for overdamped drift + diffusion on consecutive increments.

Model (1D, natural times between observations):

    ΔX_k | v, D ~ Normal(v δt_k, 2 D δt_k),   k = 0 … N−1

with independent increments (non-overlapping steps). Inference uses **PyMC** (NUTS)
and **ArviZ** for summaries and plots — the standard stack for this Gaussian
likelihood; no custom MCMC.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy import stats


def consecutive_increments(
    x: np.ndarray,
    t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Adjacent-step displacements δx and frame gaps δt (same units as CSV ``t``)."""
    xx = np.asarray(x, dtype=np.float64).ravel()
    tt = np.asarray(t, dtype=np.float64).ravel()
    if xx.size != tt.size or xx.size < 2:
        return np.array([]), np.array([])
    dx = np.diff(xx)
    dt = np.diff(tt)
    if np.any(dt <= 0):
        raise ValueError("times must be strictly increasing for consecutive increments")
    return dx, dt


def langevin_increment_logpdf_sum(
    dx: np.ndarray,
    dt: np.ndarray,
    v: float,
    d: float,
) -> float:
    """Sum of log pdfs Σ_k log Normal(dx_k | v δt_k, 2 D δt_k). ``D > 0`` required."""
    if not (math.isfinite(v) and math.isfinite(d) and d > 0.0):
        return float("-inf")
    scale = np.sqrt(2.0 * d * dt)
    return float(np.sum(stats.norm.logpdf(dx, loc=v * dt, scale=scale)))


def langevin_mle(dx: np.ndarray, dt: np.ndarray) -> tuple[float, float]:
    """Closed-form MLE on consecutive increments (secant drift + scaled residual variance)."""
    if dx.size < 1 or dt.size != dx.size:
        return float("nan"), float("nan")
    tsum = float(np.sum(dt))
    if tsum <= 0.0 or not math.isfinite(tsum):
        return float("nan"), float("nan")
    v = float(np.sum(dx) / tsum)
    resid = dx - v * dt
    d = float(np.mean((resid * resid) / (2.0 * dt)))
    if not (math.isfinite(d) and d > 0.0):
        return v, float("nan")
    return v, d


@dataclass(frozen=True)
class PymcPosteriorSummary:
    """Marginal posterior summaries (pixel units for ``v`` and ``D``)."""

    v_mean: float
    v_sd: float
    d_mean: float
    d_sd: float
    d_q05: float
    d_q95: float
    v_r_hat: float
    d_r_hat: float
    ess_v: float
    ess_D: float


def summarize_langevin_idata(idata: az.InferenceData) -> PymcPosteriorSummary:
    """Posterior means, SDs, quantiles, and standard MCMC diagnostics for ``v`` and ``D`` (px units)."""
    s = az.summary(
        idata,
        var_names=["v", "D"],
        kind="all",
        round_to="none",
    )
    post = idata.posterior
    v_s = np.asarray(post["v"].stack(sample=("chain", "draw")).values, dtype=np.float64).ravel()
    d_s = np.asarray(post["D"].stack(sample=("chain", "draw")).values, dtype=np.float64).ravel()
    d_q05, d_q95 = float(np.quantile(d_s, 0.05)), float(np.quantile(d_s, 0.95))

    return PymcPosteriorSummary(
        v_mean=float(s.loc["v", "mean"]),
        v_sd=float(s.loc["v", "sd"]),
        d_mean=float(s.loc["D", "mean"]),
        d_sd=float(s.loc["D", "sd"]),
        d_q05=d_q05,
        d_q95=d_q95,
        v_r_hat=float(s.loc["v", "r_hat"]),
        d_r_hat=float(s.loc["D", "r_hat"]),
        ess_v=float(s.loc["v", "ess_bulk"]),
        ess_D=float(s.loc["D", "ess_bulk"]),
    )


def fit_langevin_pymc(
    dx: np.ndarray,
    dt: np.ndarray,
    *,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    cores: int = 1,
    target_accept: float = 0.92,
    sigma_v_prior: float = 50.0,
    pixel_um: float | None = None,
    dt_s: float | None = None,
    random_seed: int | None = None,
    progressbar: bool = False,
) -> az.InferenceData:
    """NUTS posterior for ``v`` (px/frame) and ``D`` (px²/frame).

    Likelihood: ``dx_k ~ Normal(v * dt_k, sqrt(2 * D * dt_k))``.

    Optional calibration adds deterministic ``v_um_s`` and ``D_um2_s`` for plots/IO.

    Parameters
    ----------
    draws, tune, chains :
        Standard ``pm.sample`` settings.
    cores :
        Parallel workers (``1`` avoids multiprocessing overhead / Windows quirks for small models).
    target_accept :
        NUTS step size adaptation (higher → smaller step, safer for difficult geometries).
    """
    dx_obs = np.asarray(dx, dtype=np.float64).ravel()
    dt_obs = np.asarray(dt, dtype=np.float64).ravel()
    if dx_obs.size < 2 or dt_obs.size != dx_obs.size:
        raise ValueError("need at least two consecutive increments")
    v_mle, d_mle = langevin_mle(dx_obs, dt_obs)
    if not (math.isfinite(v_mle) and math.isfinite(d_mle) and d_mle > 0.0):
        raise ValueError("invalid MLE for initvals; check increments")

    d_scale = float(max(d_mle, 1e-12) * 15.0)

    _pm_log = logging.getLogger("pymc")
    _pm_prev = _pm_log.level
    if not progressbar:
        _pm_log.setLevel(logging.ERROR)
    try:
        with pm.Model() as model:
            v = pm.Normal("v", mu=0.0, sigma=float(sigma_v_prior))
            D = pm.HalfNormal("D", sigma=d_scale)
            mu = v * dt_obs
            sigma = pt.sqrt(pt.maximum(2.0 * D * dt_obs, 1e-20))
            pm.Normal("dx_obs", mu=mu, sigma=sigma, observed=dx_obs)

            if pixel_um is not None and dt_s is not None:
                pu, ds = float(pixel_um), float(dt_s)
                pm.Deterministic("v_um_s", v * pu / ds)
                pm.Deterministic("D_um2_s", D * (pu**2) / ds)

            idata: az.InferenceData = pm.sample(
                draws=int(draws),
                tune=int(tune),
                chains=int(chains),
                cores=int(cores),
                target_accept=float(target_accept),
                random_seed=random_seed,
                return_inferencedata=True,
                progressbar=progressbar,
                initvals={"v": v_mle, "D": d_mle},
            )
    finally:
        if not progressbar:
            _pm_log.setLevel(_pm_prev)

    return idata
