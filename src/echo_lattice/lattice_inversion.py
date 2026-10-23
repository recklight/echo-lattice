"""Room size, source distance and playback latency from the image-source lattice of one response.

The first-order inversion (:mod:`echo_lattice.inversion`) needs
a synchronized recording and an assumed source bearing, and it only looks at
nine arrivals. This module fits the whole lattice of image sources of a
rectangular room (2-D or 3-D) to every echo that was picked, with four
unknowns the first-order solution has to assume away:

* **playback latency** ``delta``: a recording whose time origin is not the
  emission instant (another device played the sound, or the converters add
  a delay). For two images ``A`` and ``B`` whose reflections lie on
  different axes, their combination ``A+B`` is also an image and the true
  path lengths satisfy ``|A+B|^2 + |0|^2 = |A|^2 + |B|^2`` (``0`` is the
  direct path). Measured lengths are ``r = q + delta``, and the squares of
  ``delta`` cancel, so every such triple gives ``delta`` in closed form:
  ``delta = (r_AB^2 + r_0^2 - r_A^2 - r_B^2) / (2 (r_AB + r_0 - r_A - r_B))``.
* **source bearing**: first-order arrivals leave it free (see
  :func:`~echo_lattice.inversion.bearing_family`); the images
  that bounce twice along one axis, at ``2L - c`` and ``2L + c`` from the
  microphone along that axis, fix the source offset ``c`` of every axis.
* **which echo is which**: no labels are assigned. The detected echoes are
  scored as a Poisson point process whose intensity has one Gaussian bump
  per predicted image (weighted by how likely that image is to be
  detected) on top of a uniform clutter rate, and the expected number of
  detections is subtracted, so a geometry that predicts strong echoes that
  were never picked is penalized as much as one that leaves echoes
  unexplained.
* **missing and spurious peaks**: the clutter term absorbs spurious peaks,
  the detection probability absorbs missing ones.

Search runs in four steps: latency hypotheses from the identity above;
wall hypotheses per axis (single-axis image groups found through the same
identity, and a per-axis scan of echo pairs over a grid of the
source-offset component); the closure ``sum(c_i^2) = q_0^2`` that couples
the axes; and an expectation-maximization fit with a shrinking kernel
width, followed by a one-dimension-at-a-time rescan of each room side. The
standard errors come from the Laplace approximation at the optimum.

All lengths are path lengths in meters, so the module does not depend on
the sampling rate except through ``resolution_m`` (arrivals closer than
that merge into one peak).
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from .errors import InversionError

_LOGGER = logging.getLogger(__name__)

#: Hypothesis generators understood by :func:`invert_lattice`.
SEARCH_METHODS: tuple[str, ...] = ("image_groups", "axis_scan")

#: Small corrections tried around every latency hypothesis by the axis scan (meters).
_LATENCY_SHIFTS_M: NDArray[np.float64] = np.arange(-0.04, 0.0401, 0.02)

#: Kernel widths of the expectation-maximization fit, widest first (meters).
FIT_WIDTHS_M: tuple[float, ...] = (0.06, 0.03, 0.015, 0.008)

_EPS = 1e-9


@dataclass(frozen=True)
class EchoModel:
    """How image arrivals become picked peaks.

    Attributes
    ----------
    relative_floor : float
        Peaks weaker than this fraction of the direct sound are not picked.
    reflection : float
        Mean wall reflection coefficient assumed when predicting which
        images are strong enough to be picked.
    clutter : float
        Expected number of spurious peaks in the analyzed window.
    max_detection : float
        Probability that a strong, isolated arrival is picked.
    resolution_m : float
        Arrivals closer than this (path length) form a single peak; one
        sample at the analysis rate.

    Examples
    --------
    >>> EchoModel().relative_floor, round(EchoModel(resolution_m=343 / 48_000).resolution_m, 4)
    (0.01, 0.0071)
    """

    relative_floor: float = 0.01
    reflection: float = 0.8
    clutter: float = 3.0
    max_detection: float = 0.9
    resolution_m: float = 343.0 / 16_000

    def __post_init__(self) -> None:
        if not 0 < self.relative_floor < 1:
            raise ValueError(f"relative_floor must lie in (0, 1), got {self.relative_floor!r}")
        if not 0 < self.reflection <= 1:
            raise ValueError(f"reflection must lie in (0, 1], got {self.reflection!r}")
        if not self.clutter > 0:
            raise ValueError(f"clutter must be positive, got {self.clutter!r}")
        if not 0 < self.max_detection <= 1:
            raise ValueError(f"max_detection must lie in (0, 1], got {self.max_detection!r}")
        if not self.resolution_m > 0:
            raise ValueError(f"resolution_m must be positive, got {self.resolution_m!r}")


@dataclass(frozen=True)
class LatticeEstimate:
    """The fitted rectangular room.

    Sides are sorted, longest first. One omnidirectional microphone cannot
    tell which side is vertical, so in 3-D ``height_m`` is only the
    shortest side. Positions are one of the layouts that produce the same
    response: per axis the source is placed after the microphone
    (``c = source - microphone >= 0``) and the microphone is no farther from
    the low wall than the source is from the high wall.

    Attributes
    ----------
    size_m : tuple of float
        Room sides, longest first (two or three values).
    microphone_m, source_m : tuple of float
        Positions on the same sorted axes.
    latency_m : float
        Playback latency as a path length (``latency_ms`` gives the time).
    size_sd_m : tuple of float
        Standard error of each side (Laplace approximation).
    latency_sd_m : float
        Standard error of the latency.
    source_distance_m : float
        Microphone-source distance.
    bearing_deg : tuple of float
        Direction of the source seen from the microphone on the sorted axes:
        ``(angle from the longest side,)`` in 2-D, ``(azimuth, elevation)``
        in 3-D, each in ``[0, 90]``.
    log_likelihood : float
        Point-process log-likelihood of the picked echoes.
    n_echoes : int
        Echoes used, the direct sound included.
    explained_fraction : float
        Share of the picked echoes that sit on a predicted arrival.
    missed_fraction : float
        Share of the strong predicted arrivals with no picked echo.
    reliable : bool
        Whether both shares pass the thresholds given to :func:`invert_lattice`.

    Examples
    --------
    >>> estimate = LatticeEstimate((6.0, 4.0), (1.0, 1.2), (2.0, 2.2), 0.3, (0.002, 0.002),
    ...                            0.001, 1.414, (45.0,), -40.0, 25, 0.92, 0.05, True)
    >>> estimate.width_m, estimate.height_m, round(estimate.latency_ms(343.0), 3)
    (6.0, None, 0.875)
    """

    size_m: tuple[float, ...]
    microphone_m: tuple[float, ...]
    source_m: tuple[float, ...]
    latency_m: float
    size_sd_m: tuple[float, ...]
    latency_sd_m: float
    source_distance_m: float
    bearing_deg: tuple[float, ...]
    log_likelihood: float
    n_echoes: int
    explained_fraction: float
    missed_fraction: float
    reliable: bool

    @property
    def width_m(self) -> float:
        """Longest side."""
        return self.size_m[0]

    @property
    def depth_m(self) -> float:
        """Second-longest side."""
        return self.size_m[1]

    @property
    def height_m(self) -> float | None:
        """Shortest side of a 3-D room (``None`` in 2-D)."""
        return self.size_m[2] if len(self.size_m) == 3 else None

    def latency_ms(self, speed_of_sound_m_s: float) -> float:
        """Latency in milliseconds."""
        return 1000.0 * self.latency_m / speed_of_sound_m_s

    def as_row(self, speed_of_sound_m_s: float) -> dict[str, Any]:
        """Flat columns for a results table (3-D columns are ``nan`` in 2-D).

        Examples
        --------
        >>> estimate = LatticeEstimate((6.0, 4.0), (1.0, 1.2), (2.0, 2.2), 0.343, (0.002, 0.002),
        ...                            0.001, 1.414, (45.0,), -40.0, 25, 0.92, 0.05, True)
        >>> row = estimate.as_row(343.0)
        >>> row["width_m"], row["latency_ms"], row["height_m"]
        (6.0, 1.0, nan)
        """
        nan = float("nan")
        padded = list(self.size_m) + [nan] * (3 - len(self.size_m))
        sd = list(self.size_sd_m) + [nan] * (3 - len(self.size_sd_m))
        mic = list(self.microphone_m) + [nan] * (3 - len(self.microphone_m))
        src = list(self.source_m) + [nan] * (3 - len(self.source_m))
        return {
            "width_m": padded[0],
            "depth_m": padded[1],
            "height_m": padded[2],
            "source_distance_m": self.source_distance_m,
            "latency_ms": self.latency_ms(speed_of_sound_m_s),
            "azimuth_deg": self.bearing_deg[0],
            "elevation_deg": self.bearing_deg[1] if len(self.bearing_deg) > 1 else nan,
            "mic_x_m": mic[0],
            "mic_y_m": mic[1],
            "mic_z_m": mic[2],
            "source_x_m": src[0],
            "source_y_m": src[1],
            "source_z_m": src[2],
            "width_sd_m": sd[0],
            "depth_sd_m": sd[1],
            "height_sd_m": sd[2],
            "latency_sd_ms": 1000.0 * self.latency_sd_m / speed_of_sound_m_s,
            "log_likelihood": self.log_likelihood,
            "explained_fraction": self.explained_fraction,
            "missed_fraction": self.missed_fraction,
            "reliable": self.reliable,
        }

    def as_dict(self) -> dict[str, Any]:
        """Plain values, ready for JSON."""
        return {
            "size_m": list(self.size_m),
            "microphone_m": list(self.microphone_m),
            "source_m": list(self.source_m),
            "latency_m": self.latency_m,
            "size_sd_m": list(self.size_sd_m),
            "latency_sd_m": self.latency_sd_m,
            "source_distance_m": self.source_distance_m,
            "bearing_deg": list(self.bearing_deg),
            "log_likelihood": self.log_likelihood,
            "n_echoes": self.n_echoes,
            "explained_fraction": self.explained_fraction,
            "missed_fraction": self.missed_fraction,
            "reliable": self.reliable,
        }


# ------------------------------------------------------------------------ lattice
class _Lattice:
    """Image indices up to ``max_order`` per axis and their path lengths.

    Parameters are packed as ``theta = (L_1..L_D, m_1..m_D, s_1..s_D, delta)``.
    """

    def __init__(self, n_dims: int, max_order: int) -> None:
        ks = np.arange(-max_order, max_order + 1)
        index = np.stack(np.meshgrid(*([ks] * n_dims), indexing="ij"), -1).reshape(-1, n_dims)
        odd = np.mod(index, 2)
        self.n_dims = n_dims
        self.sign = np.where(odd == 1, -1.0, 1.0)
        self.mult = (index + odd).astype(np.float64)
        self.n_reflections = np.abs(index).sum(axis=1)

    def lengths(
        self, theta: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Measured path lengths (latency included) and true distances of every image."""
        D = self.n_dims
        size, mic, src, delta = theta[:D], theta[D : 2 * D], theta[2 * D : 3 * D], theta[-1]
        disp = self.sign * src + self.mult * size - mic
        distance = np.linalg.norm(disp, axis=1)
        return distance + delta, distance

    def batch(self, thetas: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """:meth:`lengths` for every row of ``thetas`` (shape ``(G, 3D + 1)``)."""
        D = self.n_dims
        size, mic, src = thetas[:, :D], thetas[:, D : 2 * D], thetas[:, 2 * D : 3 * D]
        disp = (
            self.sign[None] * src[:, None, :] + self.mult[None] * size[:, None, :] - mic[:, None, :]
        )
        distance = np.linalg.norm(disp, axis=2)
        return distance + thetas[:, -1:], distance

    def jacobian(
        self, theta: NDArray[np.float64], images: NDArray[np.int64]
    ) -> NDArray[np.float64]:
        """Derivative of the measured path lengths of ``images`` with respect to ``theta``."""
        D = self.n_dims
        size, mic, src = theta[:D], theta[D : 2 * D], theta[2 * D : 3 * D]
        disp = self.sign[images] * src + self.mult[images] * size - mic
        unit = disp / np.maximum(np.linalg.norm(disp, axis=1, keepdims=True), _EPS)
        J = np.empty((len(images), 3 * D + 1))
        J[:, :D] = unit * self.mult[images]
        J[:, D : 2 * D] = -unit
        J[:, 2 * D : 3 * D] = unit * self.sign[images]
        J[:, -1] = 1.0
        return J


def image_path_lengths(
    size_m: ArrayLike, microphone_m: ArrayLike, source_m: ArrayLike, *, max_order: int = 3
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Path lengths of every image source up to ``max_order`` per axis, shortest first.

    Parameters
    ----------
    size_m, microphone_m, source_m : array_like, shape (2,) or (3,)
        Room sides and positions.
    max_order : int, optional
        Largest image index per axis.

    Returns
    -------
    lengths : numpy.ndarray
        Microphone-image distances in meters, increasing.
    n_reflections : numpy.ndarray
        Wall reflections along each path.

    Examples
    --------
    >>> lengths, bounces = image_path_lengths([6.0, 4.0], [2.2, 1.3], [3.2, 2.3])
    >>> lengths[:3].round(3).tolist(), bounces[:3].tolist()
    ([1.414, 3.736, 4.512], [0, 1, 1])
    """
    size = np.asarray(size_m, dtype=np.float64).ravel()
    lattice = _Lattice(size.size, int(max_order))
    theta = np.concatenate(
        [size, np.asarray(microphone_m, float).ravel(), np.asarray(source_m, float).ravel(), [0.0]]
    )
    _, distance = lattice.lengths(theta)
    order = np.argsort(distance, kind="stable")
    return distance[order], lattice.n_reflections[order]


# --------------------------------------------------------------------- likelihood
def detection_probability(
    n_reflections: ArrayLike, distance_m: ArrayLike, direct_m: ArrayLike, model: EchoModel
) -> NDArray[np.float64]:
    """Probability that an image arrival is picked.

    The predicted level relative to the direct sound is
    ``reflection ** n_reflections * direct_m / distance_m``; the probability
    rises smoothly around ``relative_floor`` (a 4 dB wide logistic step) up
    to ``max_detection``.

    Examples
    --------
    >>> detection_probability([0, 1, 6], [1.0, 3.0, 40.0], 1.0, EchoModel()).round(2).tolist()
    [0.9, 0.9, 0.12]
    """
    rel = model.reflection ** np.asarray(n_reflections, dtype=np.float64)
    rel = rel * np.asarray(direct_m, dtype=np.float64) / np.maximum(np.asarray(distance_m), _EPS)
    level_db = 20.0 * np.log10(np.maximum(rel, 1e-12) / model.relative_floor)
    return model.max_detection / (1.0 + np.exp(-level_db / 2.0))


def _window(observed: NDArray[np.float64], model: EchoModel) -> tuple[float, float, float]:
    low = observed[0] - 3 * model.resolution_m
    high = observed[-1] + 3 * model.resolution_m
    return low, high, high - low


def _merged_expectation(
    lengths: NDArray[np.float64], prob: NDArray[np.float64], gap: float
) -> NDArray[np.float64]:
    """Expected number of picked peaks per row: arrivals closer than ``gap`` count once."""
    order = np.argsort(lengths, axis=-1)
    ts = np.take_along_axis(lengths, order, axis=-1)
    ps = np.take_along_axis(prob, order, axis=-1)
    flat_t = ts.reshape(-1, ts.shape[-1])
    flat_p = ps.reshape(-1, ps.shape[-1])
    new = np.ones_like(flat_t, dtype=bool)
    new[:, 1:] = np.diff(flat_t, axis=1) >= gap
    starts = np.flatnonzero(new.ravel())
    peaks = np.maximum.reduceat(flat_p.ravel(), starts)
    rows = starts // flat_t.shape[1]
    out = np.bincount(rows, weights=peaks, minlength=flat_t.shape[0])
    return out.reshape(ts.shape[:-1])


def point_process_loglik(
    observed_m: ArrayLike,
    predicted_m: ArrayLike,
    probability: ArrayLike,
    width_m: float,
    model: EchoModel,
) -> float:
    """Log-likelihood of the picked echoes under one predicted set of arrivals.

    Echoes form a Poisson point process with intensity
    ``clutter / span + sum_i p_i N(r; rho_i, width^2)`` over the analyzed
    span; the expected number of picked peaks, ``clutter + sum_i p_i``
    (merged arrivals counted once), is subtracted.

    Parameters
    ----------
    observed_m : array_like
        Picked echoes as path lengths, increasing.
    predicted_m, probability : array_like
        Predicted path lengths and their detection probabilities.
    width_m : float
        Standard deviation of an echo around its predicted position.
    model : EchoModel
        Clutter rate and peak resolution.

    Examples
    --------
    >>> m = EchoModel()
    >>> good = point_process_loglik([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [0.9] * 3, 0.01, m)
    >>> bad = point_process_loglik([1.0, 2.0, 3.0], [1.0, 2.5, 3.5], [0.9] * 3, 0.01, m)
    >>> good > bad
    True
    """
    r = np.sort(np.asarray(observed_m, dtype=np.float64))
    rho = np.asarray(predicted_m, dtype=np.float64)
    prob = np.asarray(probability, dtype=np.float64)
    low, high, span = _window(r, model)
    inside = (rho > low) & (rho < high)
    rho, prob = rho[inside], prob[inside]
    density = np.exp(-0.5 * ((r[:, None] - rho[None, :]) / width_m) ** 2)
    density = density @ prob / (width_m * np.sqrt(2 * np.pi))
    expected = (
        float(_merged_expectation(rho[None, :], prob[None, :], model.resolution_m)[0])
        if rho.size
        else 0.0
    )
    return float(np.log(model.clutter / span + density).sum() - expected - model.clutter)


def _batch_loglik(
    r: NDArray[np.float64],
    rho: NDArray[np.float64],
    prob: NDArray[np.float64],
    width_m: float,
    model: EchoModel,
) -> NDArray[np.float64]:
    """:func:`point_process_loglik` for many rows; each echo sees its two nearest predictions."""
    G = rho.shape[0]
    low, high, span = _window(r, model)
    inside = (rho > low) & (rho < high)
    big = 1e7
    masked = np.where(inside, rho, big)
    order = np.argsort(masked, axis=1)
    ts = np.take_along_axis(masked, order, axis=1)
    ps = np.take_along_axis(np.where(inside, prob, 0.0), order, axis=1)
    expected = _merged_expectation(ts, ps, model.resolution_m)
    offset = np.arange(G)[:, None] * 4 * big
    flat = (ts + offset).ravel()
    pflat = ps.ravel()
    query = (r[None, :] + offset).ravel()
    pos = np.searchsorted(flat, query)
    lo = np.clip(pos - 1, 0, flat.size - 1)
    hi = np.clip(pos, 0, flat.size - 1)
    near = np.exp(-0.5 * ((query - flat[lo]) / width_m) ** 2) * pflat[lo]
    near += np.exp(-0.5 * ((query - flat[hi]) / width_m) ** 2) * pflat[hi]
    near /= width_m * np.sqrt(2 * np.pi)
    loglik = np.log(model.clutter / span + near.reshape(G, -1)).sum(axis=1)
    return loglik - expected - model.clutter


# ------------------------------------------------------------------ latency search
def latency_candidates(
    observed_m: ArrayLike,
    *,
    max_candidates: int = 8,
    early: int = 12,
    tolerance_m: float = 0.015,
    min_denominator_m: float = 0.2,
) -> list[float]:
    """Playback-latency hypotheses from triples of echoes.

    Every pair ``(A, B)`` among the ``early`` first echoes and every later
    echo taken as ``A+B`` gives a latency in closed form (module docstring).
    Each hypothesis is then scored by how many early pairs have an echo
    where their combination should be, and the best distinct ones (at least
    8 cm apart) are returned.

    Parameters
    ----------
    observed_m : array_like
        Picked echoes as path lengths; the shortest is the direct sound.
    max_candidates : int, optional
        Hypotheses returned.
    early : int, optional
        Echoes (the direct sound included) used as ``A`` and ``B``.
    tolerance_m : float, optional
        Match tolerance of the scoring.
    min_denominator_m : float, optional
        Triples with a smaller denominator are too ill-conditioned to use.

    Returns
    -------
    list of float
        Latencies in meters of path length, best first.

    Examples
    --------
    >>> lengths, _ = image_path_lengths([6.0, 4.0], [2.2, 1.3], [3.2, 2.3])
    >>> round(latency_candidates(lengths[:25] + 2.0)[0], 3)
    2.0
    """
    r = np.sort(np.asarray(observed_m, dtype=np.float64))
    n = r.size
    M = min(int(early), n)
    if M < 3:
        return []
    r0 = r[0]
    hypotheses = []
    for i, j in itertools.combinations(range(1, M), 2):
        k = np.arange(j + 1, n)
        den = r[k] + r0 - r[i] - r[j]
        num = r[k] ** 2 + r0**2 - r[i] ** 2 - r[j] ** 2
        good = np.abs(den) > min_denominator_m
        hypotheses.append(num[good] / (2 * den[good]))
    if not hypotheses:
        return []
    h = np.concatenate(hypotheses)
    h = np.unique(np.round(h[(h > -5.0) & (h < r0 - 0.2)], 4))
    if h.size == 0:
        return []
    ii, jj = np.triu_indices(M - 1, 1)
    ii, jj = ii + 1, jj + 1
    scores = np.empty(h.size)
    for row, delta in enumerate(h):
        q = r - delta
        p = np.sqrt(np.maximum(q[ii] ** 2 + q[jj] ** 2 - q[0] ** 2, 0.0))
        scores[row] = np.count_nonzero(_nearest_gap(q, p) < tolerance_m)
    picked: list[float] = []
    for row in np.argsort(-scores, kind="stable"):
        if all(abs(h[row] - x) > 0.08 for x in picked):
            picked.append(float(h[row]))
        if len(picked) == max_candidates:
            break
    return picked


def _nearest_gap(
    sorted_values: NDArray[np.float64], query: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Distance from every query value to the nearest entry of ``sorted_values``."""
    n = sorted_values.size
    pos = np.searchsorted(sorted_values, query)
    lo = sorted_values[np.clip(pos - 1, 0, n - 1)]
    hi = sorted_values[np.clip(pos, 0, n - 1)]
    return np.minimum(np.abs(query - lo), np.abs(query - hi))


# ----------------------------------------------------- walls to geometry (batched)
def _geometry(
    walls: NDArray[np.float64], q0: float, delta: float, unit: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Room parameters from per-axis wall lengths and a bearing.

    ``walls`` has shape ``(G, D, 2)`` (the two first-order wall paths of
    every axis, latency removed) and ``unit`` shape ``(G, D)`` (unit source
    direction, non-negative). Returns ``theta`` rows and their validity.
    """
    comp = q0 * unit
    rho2 = q0**2 - comp**2
    with np.errstate(invalid="ignore"):
        Ua = np.sqrt(walls[..., 0] ** 2 - rho2)
        Ub = np.sqrt(walls[..., 1] ** 2 - rho2)
    L = 0.5 * (Ua + Ub)
    ma, mb = 0.5 * (Ua - comp), 0.5 * (Ub - comp)
    va = (ma > 0) & (ma + comp < L)
    vb = (mb > 0) & (mb + comp < L)
    mic = np.where(va, ma, mb)
    valid = np.all((va | vb) & np.isfinite(L), axis=1)
    thetas = np.concatenate([L, mic, mic + comp, np.full((L.shape[0], 1), delta)], axis=1)
    return np.nan_to_num(thetas, nan=1.0), valid


class _Scorer:
    """Batched point-process scores of parameter rows against one set of echoes."""

    def __init__(self, r: NDArray[np.float64], lattice: _Lattice, model: EchoModel, width_m: float):
        self.r, self.lattice, self.model, self.width_m = r, lattice, model, width_m

    def __call__(
        self, thetas: NDArray[np.float64], valid: NDArray[np.bool_] | None = None
    ) -> NDArray[np.float64]:
        if thetas.shape[0] == 0:
            return np.zeros(0)
        D = self.lattice.n_dims
        rho, distance = self.lattice.batch(thetas)
        q0 = np.linalg.norm(thetas[:, 2 * D : 3 * D] - thetas[:, D : 2 * D], axis=1)
        prob = detection_probability(
            self.lattice.n_reflections[None, :], distance, q0[:, None], self.model
        )
        scores = _batch_loglik(self.r, rho, prob, self.width_m, self.model)
        if valid is not None:
            scores = np.where(valid, scores, -np.inf)
        return scores


def bearing_grid(n_dims: int, step_deg: float) -> NDArray[np.float64]:
    """Unit source directions with non-negative components, ``step_deg`` apart.

    Examples
    --------
    >>> bearing_grid(2, 45.0).round(3).tolist()
    [[1.0, 0.0], [0.707, 0.707], [0.0, 1.0]]
    >>> bearing_grid(3, 45.0).shape
    (7, 3)
    """
    angles = np.deg2rad(np.arange(0.0, 90.0 + 1e-9, step_deg))
    if n_dims == 2:
        return np.stack([np.cos(angles), np.sin(angles)], axis=1)
    az, el = np.meshgrid(angles, angles, indexing="ij")
    unit = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=-1)
    return np.unique(np.round(unit.reshape(-1, 3), 9), axis=0)


# ---------------------------------------------------------- search: axis scan
_KINDS = ("walls", "minus", "plus")


def _axis_table(
    q: NDArray[np.float64],
    first: NDArray[np.int64],
    second: NDArray[np.int64],
    components: NDArray[np.float64],
    width_m: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Score every echo pair as the evidence that fixes one axis, for each offset ``c``.

    A pair ``(a, b)`` is read as the two first-order walls, or as one wall
    plus the image at ``2L - c`` (``minus``) or ``2L + c`` (``plus``); that
    fixes ``L`` and the microphone coordinate, and the pair is scored by the
    other single-axis images (``|k| <= 3``) found among the echoes. The
    score does not depend on which axis it is, so one table serves all
    axes. Returns scores and the implied wall lengths, shape ``(n_c, 3 H)``.
    """
    q0 = q[0]
    A, B = np.meshgrid(first, second, indexing="ij")
    keep = A != B
    qa, qb_obs = q[A[keep]][None, :], q[B[keep]][None, :]
    c = np.asarray(components, dtype=np.float64)[:, None]
    rho2 = q0**2 - c**2
    with np.errstate(invalid="ignore"):
        Ua = np.sqrt(qa**2 - rho2)
        Db = np.sqrt(qb_obs**2 - rho2)
    scores, wall_b = [], []
    for kind in _KINDS:
        if kind == "walls":
            L, Ub, qb, used = 0.5 * (Ua + Db), Db, np.broadcast_to(qb_obs, Db.shape), (0, 1)
        else:
            L = 0.5 * (Db + c) if kind == "minus" else 0.5 * (Db - c)
            Ub = 2 * L - Ua
            with np.errstate(invalid="ignore"):
                qb = np.sqrt(Ub**2 + rho2)
            used = (0, 2) if kind == "minus" else (0, 3)
        m = 0.5 * (Ua - c)
        valid = (m > 0) & (m + c < L) & (Ub > 0) & np.isfinite(L)
        offsets = np.stack([Ua, Ub, 2 * L - c, 2 * L + c, 2 * L + Ua, 2 * L + Ub], axis=-1)
        mask = np.ones(6, dtype=bool)
        mask[list(used)] = False
        with np.errstate(invalid="ignore"):
            predicted = np.sqrt(offsets[..., mask] ** 2 + rho2[..., None])
        gap = _nearest_gap(q, np.nan_to_num(predicted, nan=1e6).ravel()).reshape(predicted.shape)
        score = np.exp(-0.5 * (gap / width_m) ** 2).sum(axis=-1)
        if kind != "walls":
            score = score - 0.25  # two observed walls win a tie
        scores.append(np.where(valid, score, -np.inf))
        wall_b.append(qb)
    wall_a = np.broadcast_to(np.tile(qa, (1, len(_KINDS))), (c.shape[0], qa.shape[1] * len(_KINDS)))
    return np.concatenate(scores, axis=1), wall_a, np.concatenate(wall_b, axis=1)


def _top_pairs(
    score: NDArray[np.float64], qa: NDArray[np.float64], qb: NDArray[np.float64], k: int
) -> list[tuple[float, float]]:
    picked: list[tuple[float, float]] = []
    for row in np.argsort(-score):
        if not np.isfinite(score[row]) or len(picked) == k:
            break
        if all(abs(qa[row] - a) + abs(qb[row] - b) > 0.02 for a, b in picked):
            picked.append((float(qa[row]), float(qb[row])))
    return picked


def _search_axis_scan(
    r: NDArray[np.float64],
    delta: float,
    n_dims: int,
    scorer: _Scorer,
    unit_grid: NDArray[np.float64],
    *,
    n_first: int = 20,
    per_axis: int = 4,
    n_components: int = 151,
    shifts_m: NDArray[np.float64] = _LATENCY_SHIFTS_M,
    keep_shifts: int = 2,
    width_m: float = 0.02,
) -> list[tuple[float, NDArray[np.float64]]]:
    first = np.arange(1, min(n_first + 1, r.size))
    second = np.arange(1, r.size)
    tables = []
    for shift in shifts_m:
        q = r - (delta + shift)
        if q[0] <= 0.1:
            continue
        S, QA, QB = _axis_table(q, first, second, np.linspace(0.0, q[0], n_components), width_m)
        tables.append((float(delta + shift), float(q[0]), S, QA, QB, S.max(axis=1)))
    rows_w, rows_u, rows_meta = [], [], []
    for unit in unit_grid:
        index = np.rint(unit * (n_components - 1)).astype(int)
        ranked = sorted(tables, key=lambda t: -float(np.sum(t[5][index])))
        for lat, q0, S, QA, QB, smax in ranked[:keep_shifts]:
            if not np.isfinite(np.sum(smax[index])):
                continue
            tops = [_top_pairs(S[i], QA[i], QB[i], per_axis) for i in index]
            for combo in itertools.product(*tops):
                rows_w.append(combo)
                rows_u.append(unit)
                rows_meta.append((q0, lat))
    return _score_rows(rows_w, rows_u, rows_meta, n_dims, scorer)


def _score_rows(
    walls: list[Any],
    units: list[NDArray[np.float64]],
    meta: list[tuple[float, float]],
    n_dims: int,
    scorer: _Scorer,
    chunk: int = 4096,
) -> list[tuple[float, NDArray[np.float64]]]:
    """Geometry and score of every (walls, bearing, q0, latency) row, in chunks."""
    out: list[tuple[float, NDArray[np.float64]]] = []
    for start in range(0, len(walls), chunk):
        w = np.asarray(walls[start : start + chunk], dtype=np.float64).reshape(-1, n_dims, 2)
        u = np.asarray(units[start : start + chunk], dtype=np.float64)
        q0 = np.array([m[0] for m in meta[start : start + chunk]])
        lat = np.array([m[1] for m in meta[start : start + chunk]])
        thetas = np.empty((w.shape[0], 3 * n_dims + 1))
        valid = np.empty(w.shape[0], dtype=bool)
        for value in np.unique(np.round(q0, 9)):
            rows = np.isclose(q0, value)
            th, ok = _geometry(w[rows], float(value), 0.0, u[rows])
            thetas[rows] = th
            valid[rows] = ok
        thetas[:, -1] = lat
        scores = scorer(thetas, valid)
        out.extend((float(s), t) for s, t in zip(scores, thetas, strict=True) if np.isfinite(s))
    return out


# ------------------------------------------------------- search: image groups
def _identity_graph(q: NDArray[np.float64], tolerance_m: float) -> NDArray[np.bool_]:
    """``E[a, b]``: an echo sits where the combination of images ``a`` and ``b`` would."""
    q0 = q[0]
    sq = q**2
    combo = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - q0**2, 0.0))
    graph = _nearest_gap(q, combo.ravel()).reshape(combo.shape) < tolerance_m
    graph[0, :] = False
    graph[:, 0] = False
    np.fill_diagonal(graph, False)
    return graph


def _has_echo(q: NDArray[np.float64], value: float, tolerance_m: float) -> bool:
    return bool(_nearest_gap(q, np.array([value]))[0] < tolerance_m)


def _seeds(
    q: NDArray[np.float64],
    graph: NDArray[np.bool_],
    n_dims: int,
    tolerance_m: float,
    early: int = 14,
    n_seeds: int = 6,
) -> list[tuple[int, ...]]:
    """One near wall per axis: early echoes whose pairwise combinations were picked."""
    q0 = q[0]
    M = min(early, q.size - 1)
    found = []
    if n_dims == 3:
        for a, b, c in itertools.combinations_with_replacement(range(1, M + 1), 3):
            if a == b == c:
                continue
            edges = (
                int(graph[a, b] or a == b) + int(graph[a, c] or a == c) + int(graph[b, c] or b == c)
            )
            apex = _has_echo(
                q, np.sqrt(max(q[a] ** 2 + q[b] ** 2 + q[c] ** 2 - 2 * q0**2, 0.0)), tolerance_m
            )
            if edges == 3 or (edges == 2 and apex):
                distinct = len({a, b, c}) == 3
                found.append((int(distinct), edges + int(apex), -(q[a] + q[b] + q[c]), (a, b, c)))
    else:
        for a, b in itertools.combinations(range(1, M + 1), 2):
            if graph[a, b]:
                found.append((1, 1, -(q[a] + q[b]), (a, b)))
    found.sort(reverse=True)
    return [item[-1] for item in found[:n_seeds]]


def _axis_groups(
    q: NDArray[np.float64],
    graph: NDArray[np.bool_],
    seed: tuple[int, ...],
    min_fraction: float = 0.3,
) -> list[list[int]]:
    """Echoes that behave like single-axis images of each axis.

    A single-axis image of axis ``x`` combines with every single-axis image
    of the other axes, so it is adjacent to the other seeds; members that
    are not adjacent to most of the other groups are dropped.
    """
    n_dims = len(seed)
    groups = []
    for axis in range(n_dims):
        member = np.ones(q.size, dtype=bool)
        for other in range(n_dims):
            if other != axis:
                member &= graph[:, seed[other]]
        member[0] = False
        member[seed[axis]] = True
        groups.append(set(np.flatnonzero(member).tolist()))
    cleaned = []
    for axis in range(n_dims):
        rest = set().union(*(groups[k] for k in range(n_dims) if k != axis))
        keep = []
        for echo in groups[axis]:
            others = [g for g in rest if g != echo]
            if echo == seed[axis] or (others and np.mean(graph[echo, others]) >= min_fraction):
                keep.append(echo)
        cleaned.append(sorted(keep, key=lambda i: q[i]))
    return cleaned


def _axis_fits(
    q: NDArray[np.float64],
    members: list[int],
    tolerance_m: float,
    top: int = 3,
    step_m: float = 0.005,
) -> list[tuple[float, float, float, float, float, float]]:
    """``(score, L, m, c, wall_a, wall_b)`` of one axis fitted to its single-axis images.

    The shortest member is taken as a first-order wall; the partner is one
    of the next three members, read as the other wall or as the image at
    ``2L -+ c``. Each reading is scored by the members it explains plus a
    quarter point per other echo it explains.
    """
    q0 = q[0]
    lengths = np.sort(q[members])
    if lengths.size < 2:
        return []
    c = np.arange(0.0, q0 + 1e-9, step_m)[:, None]
    rho2 = q0**2 - c**2
    D = np.sqrt(np.maximum(lengths[None, :] ** 2 - rho2, 0.0))
    U1 = D[:, :1]
    partners = D[:, 1 : min(4, lengths.size)]
    rows = []
    for kind in _KINDS:
        if kind == "walls":
            L = 0.5 * (U1 + partners)
        elif kind == "minus":
            L = 0.5 * (partners + c)
        else:
            L = 0.5 * (partners - c)
        U2 = 2 * L - U1
        m = 0.5 * (U1 - c)
        valid = (U2 > 0) & (m > 0) & (m + c < L) & (L > 0.5)
        offsets = np.stack(
            [
                np.broadcast_to(U1, L.shape),
                U2,
                2 * L - c,
                2 * L + c,
                2 * L + U1,
                2 * L + U2,
                4 * L - c,
                4 * L + c,
                4 * L + U1,
                4 * L + U2,
            ],
            axis=-1,
        )
        predicted = np.sqrt(offsets**2 + rho2[..., None])
        own = (np.abs(predicted[..., :, None] - lengths).min(axis=-2) < tolerance_m).sum(axis=-1)
        seen_gap = _nearest_gap(np.sort(q[1:]), predicted.ravel()).reshape(predicted.shape)
        seen = (seen_gap < tolerance_m).sum(axis=-1)
        score = np.where(valid, own + 0.25 * seen, -np.inf)
        cc = np.broadcast_to(c, L.shape)
        wall_a = np.sqrt(np.broadcast_to(U1, L.shape) ** 2 + rho2)
        wall_b = np.sqrt(np.maximum(U2, 0.0) ** 2 + rho2)
        rows.append(
            np.stack([score, L, np.broadcast_to(m, L.shape), cc, wall_a, wall_b], axis=-1).reshape(
                -1, 6
            )
        )
    table = np.concatenate(rows)
    table = table[np.isfinite(table[:, 0])]
    table = table[np.argsort(-table[:, 0], kind="stable")]
    distinct: list[tuple[float, float, float, float, float, float]] = []
    for fit in table:
        if all(abs(fit[1] - d[1]) > 0.03 or abs(fit[3] - d[3]) > 0.05 for d in distinct):
            distinct.append(tuple(float(x) for x in fit))
        if len(distinct) == top:
            break
    return distinct


def _walls_for_component(
    q: NDArray[np.float64], members: list[int], c: float
) -> tuple[float, float, float, float, float, float] | None:
    """An axis whose offset ``c`` comes from the closure: walls from its two shortest images."""
    q0 = q[0]
    lengths = np.sort(q[members])
    if lengths.size < 2:
        return None
    rho2 = q0**2 - c**2
    if lengths[0] ** 2 < rho2 or lengths[1] ** 2 < rho2:
        return None
    U1, U2 = np.sqrt(lengths[0] ** 2 - rho2), np.sqrt(lengths[1] ** 2 - rho2)
    L, m = 0.5 * (U1 + U2), 0.5 * (U1 - c)
    if not (m > 0 and m + c < L):
        return None
    return (0.0, float(L), float(m), float(c), float(lengths[0]), float(lengths[1]))


def _search_image_groups(
    r: NDArray[np.float64], delta: float, n_dims: int, scorer: _Scorer, tolerance_m: float
) -> list[tuple[float, NDArray[np.float64]]]:
    q = r - delta
    q0 = q[0]
    if q0 <= 0.1:
        return []
    graph = _identity_graph(q, tolerance_m)
    walls, units, meta = [], [], []

    def add(combo: list[tuple[float, float, float, float, float, float]]) -> None:
        comp = np.array([f[3] for f in combo])
        norm = float(np.linalg.norm(comp))
        if norm <= 0 or abs(norm - q0) > 0.1:
            return
        walls.append([(f[4], f[5]) for f in combo])
        units.append(comp / norm)
        meta.append((q0, delta))

    for seed in _seeds(q, graph, n_dims, tolerance_m):
        groups = _axis_groups(q, graph, seed)
        fits = [_axis_fits(q, g, 2 * tolerance_m) for g in groups]
        if all(fits):
            for combo in itertools.product(*fits):
                add(list(combo))
        for axis in range(n_dims):
            others = [fits[j] for j in range(n_dims) if j != axis]
            if not all(others):
                continue
            for part in itertools.product(*others):
                c2 = q0**2 - sum(f[3] ** 2 for f in part)
                if c2 < 0:
                    continue
                derived = _walls_for_component(q, groups[axis], float(np.sqrt(c2)))
                if derived is not None:
                    combo = list(part)
                    combo.insert(axis, derived)
                    add(combo)
    return _score_rows(walls, units, meta, n_dims, scorer)


# --------------------------------------------------------------------- fitting
def _clip_inside(theta: NDArray[np.float64], n_dims: int) -> NDArray[np.float64]:
    theta = theta.copy()
    for axis in range(n_dims):
        side = theta[axis]
        theta[n_dims + axis] = np.clip(theta[n_dims + axis], 1e-3, side - 1e-3)
        theta[2 * n_dims + axis] = np.clip(theta[2 * n_dims + axis], 1e-3, side - 1e-3)
    return theta


def fit_lattice(
    observed_m: ArrayLike,
    theta: ArrayLike,
    *,
    max_order: int = 4,
    model: EchoModel | None = None,
    widths_m: tuple[float, ...] = FIT_WIDTHS_M,
    iterations: int = 4,
) -> tuple[NDArray[np.float64], float, NDArray[np.float64]]:
    """Expectation-maximization fit of the room parameters to the picked echoes.

    Each round computes how much every echo belongs to every predicted
    arrival (the point-process responsibilities) and solves the weighted
    least-squares problem for the parameters; the kernel width shrinks
    from the first to the last value of ``widths_m``, so the fit first sees
    a smooth version of the likelihood and then the sharp one.

    Parameters
    ----------
    observed_m : array_like
        Picked echoes as path lengths.
    theta : array_like, shape (3 D + 1,)
        Start: sides, microphone, source and latency.
    max_order : int, optional
        Largest image index per axis.
    model : EchoModel, optional
        Detection model (default :class:`EchoModel`).
    widths_m : tuple of float, optional
        Kernel widths of the successive rounds, never below half the
        resolution.
    iterations : int, optional
        Rounds per width.

    Returns
    -------
    theta : numpy.ndarray
        Fitted parameters.
    log_likelihood : float
        Point-process log-likelihood at the last width.
    covariance : numpy.ndarray, shape (3 D + 1, 3 D + 1)
        Laplace approximation of the parameter covariance.

    Examples
    --------
    >>> lengths, _ = image_path_lengths([6.0, 4.0], [2.2, 1.3], [3.2, 2.3])
    >>> theta, _, cov = fit_lattice(lengths[:25] + 0.5, [6.05, 3.95, 2.25, 1.28, 3.24, 2.28, 0.52])
    >>> theta[:2].round(3).tolist(), round(float(theta[-1]), 3), cov.shape
    ([6.0, 4.0], 0.5, (7, 7))
    """
    model = model or EchoModel()
    r = np.sort(np.asarray(observed_m, dtype=np.float64))
    start = np.asarray(theta, dtype=np.float64).copy()
    n_dims = (start.size - 1) // 3
    lattice = _Lattice(n_dims, max_order)
    floor = 0.5 * model.resolution_m
    widths = tuple(max(w, floor) for w in widths_m)
    lower = np.concatenate([np.full(n_dims, 0.3), np.zeros(2 * n_dims), [-np.inf]])
    upper = np.concatenate([np.full(n_dims, 60.0), np.full(2 * n_dims, 60.0), [np.inf]])
    start = _clip_inside(np.clip(start, lower + 1e-6, upper - 1e-6), n_dims)

    def score(th: NDArray[np.float64]) -> float:
        rho, distance = lattice.lengths(th)
        q0 = float(np.linalg.norm(th[2 * n_dims : 3 * n_dims] - th[n_dims : 2 * n_dims]))
        prob = detection_probability(lattice.n_reflections, distance, q0, model)
        return point_process_loglik(r, rho, prob, widths[-1], model)

    # Two chains: the full schedule, which can walk in from a rough start, and the last two
    # widths only, which keeps a good start from being smoothed away by the wide kernels.
    # Each chain keeps the best point it visits, scored at the final width.
    best_theta, best_ll = start, score(start)
    for schedule in (widths, widths[-2:]):
        theta = start.copy()
        for width in schedule:
            theta = _em_rounds(r, theta, lattice, model, width, iterations, (lower, upper))
            theta = _polish(r, theta, lattice, model, widths[-1], (lower, upper))
            current = score(theta)
            if current > best_ll:
                best_theta, best_ll = theta.copy(), current
    return best_theta, best_ll, _laplace_covariance(r, best_theta, lattice, widths[-1], model)


def _em_rounds(
    r: NDArray[np.float64],
    theta: NDArray[np.float64],
    lattice: _Lattice,
    model: EchoModel,
    width: float,
    iterations: int,
    bounds: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Expectation-maximization rounds at one kernel width."""
    n_dims = lattice.n_dims
    low, high, span = _window(r, model)
    clutter = model.clutter / span
    for _ in range(iterations):
        rho, distance = lattice.lengths(theta)
        active = np.flatnonzero((rho > low - 4 * width) & (rho < high + 4 * width))
        if active.size == 0:
            break
        q0 = float(np.linalg.norm(theta[2 * n_dims : 3 * n_dims] - theta[n_dims : 2 * n_dims]))
        prob = detection_probability(lattice.n_reflections[active], distance[active], q0, model)
        g = np.exp(-0.5 * ((r[:, None] - rho[None, active]) / width) ** 2)
        weighted = g * prob[None, :] / (width * np.sqrt(2 * np.pi))
        resp = weighted / (clutter + weighted.sum(axis=1, keepdims=True))
        jj, ii = np.nonzero(resp > 1e-4)
        if jj.size < 3 * n_dims + 2:
            break
        weights = np.sqrt(resp[jj, ii])
        images = active[ii]
        targets = r[jj]

        def residuals(th, images=images, targets=targets, weights=weights):
            return weights * (targets - lattice.lengths(th)[0][images])

        def jacobian(th, images=images, weights=weights):
            return -weights[:, None] * lattice.jacobian(th, images)

        solution = least_squares(
            residuals, theta, jac=jacobian, method="trf", bounds=bounds, max_nfev=50
        )
        theta = _clip_inside(solution.x, n_dims)
    return theta


def _polish(
    r: NDArray[np.float64],
    theta: NDArray[np.float64],
    lattice: _Lattice,
    model: EchoModel,
    width: float,
    bounds: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Plain least squares on the echoes that have one clear partner.

    Soft responsibilities pull an arrival toward an echo that belongs to its
    neighbor, which leaves a millimeter-level bias where images crowd
    together; this removes it.
    """
    n_dims = lattice.n_dims
    low, high, _ = _window(r, model)
    for _ in range(2):
        rho, _ = lattice.lengths(theta)
        echoes, images = _clear_matches(r, rho, low, high, width)
        if echoes.size < 3 * n_dims + 2:
            break

        def plain(th, images=images, targets=r[echoes]):
            return targets - lattice.lengths(th)[0][images]

        def plain_jacobian(th, images=images):
            return -lattice.jacobian(th, images)

        solution = least_squares(
            plain, theta, jac=plain_jacobian, method="trf", bounds=bounds, max_nfev=50
        )
        theta = _clip_inside(solution.x, n_dims)
    return theta


def _clear_matches(
    r: NDArray[np.float64], rho: NDArray[np.float64], low: float, high: float, width: float
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Echoes within two widths of one predicted arrival and three widths clear of the next."""
    active = np.flatnonzero((rho > low) & (rho < high))
    if active.size < 2:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    order = active[np.argsort(rho[active])]
    sorted_rho = rho[order]
    pos = np.clip(np.searchsorted(sorted_rho, r), 1, sorted_rho.size - 1)
    left, right = sorted_rho[pos - 1], sorted_rho[pos]
    use_left = np.abs(r - left) <= np.abs(r - right)
    nearest = np.where(use_left, pos - 1, pos)
    first = np.abs(r - sorted_rho[nearest])
    second_index = np.where(
        use_left, np.clip(pos - 2, 0, None), np.clip(pos + 1, None, sorted_rho.size - 1)
    )
    other = np.where(use_left, right, left)
    beyond = np.where(second_index != nearest, np.abs(r - sorted_rho[second_index]), np.inf)
    second = np.minimum(np.abs(r - other), beyond)
    clear = (first < 2 * width) & (second > 3 * width)
    echoes = np.flatnonzero(clear)
    images = order[nearest[clear]]
    keep = np.unique(images, return_index=True)[1]  # one echo per arrival
    return echoes[keep], images[keep]


def _laplace_covariance(
    r: NDArray[np.float64],
    theta: NDArray[np.float64],
    lattice: _Lattice,
    width: float,
    model: EchoModel,
) -> NDArray[np.float64]:
    """Covariance from the echoes matched to an arrival within three kernel widths."""
    rho, _ = lattice.lengths(theta)
    low, high, _ = _window(r, model)
    active = np.flatnonzero((rho > low) & (rho < high))
    if active.size == 0:
        return np.full((theta.size, theta.size), np.nan)
    gap = np.abs(r[:, None] - rho[None, active])
    matched = np.flatnonzero(gap.min(axis=1) < 3 * width)
    images = active[np.argmin(gap[matched], axis=1)]
    J = lattice.jacobian(theta, images)
    residual = r[matched] - rho[images]
    dof = max(matched.size - theta.size, 1)
    variance = max(float(np.sum(residual**2)) / dof, (0.25 * model.resolution_m) ** 2)
    return variance * np.linalg.pinv(J.T @ J)


def _rescan_axes(
    r: NDArray[np.float64],
    theta: NDArray[np.float64],
    loglik: float,
    scorer: _Scorer,
    *,
    max_order: int,
    model: EchoModel,
    step_m: float = 0.01,
    longest_m: float = 15.0,
    sweeps: int = 2,
) -> tuple[NDArray[np.float64], float]:
    """Rescan one axis at a time and keep any clear improvement.

    Two scans per axis, everything else fixed: the side length, and the
    microphone coordinate with the source offset along that axis kept (so
    the direct path does not change). The search sometimes lands on the
    right side with the microphone pushed against a wall, when the wall
    echo it built the axis from was never picked; the second scan gets it
    out of there.
    """
    n_dims = scorer.lattice.n_dims
    for _ in range(sweeps):
        improved = False
        for axis in range(n_dims):
            side = theta[axis]
            mic, src = theta[n_dims + axis], theta[2 * n_dims + axis]
            offset = src - mic
            sides = np.arange(max(mic, src) + 0.05, longest_m, step_m)
            low = max(0.01, -offset + 0.01)
            high = min(side - 0.01, side - offset - 0.01)
            mics = np.arange(low, high, step_m)
            for values, column in ((sides, axis), (mics, n_dims + axis)):
                if values.size == 0:
                    continue
                rows = np.repeat(theta[None, :], values.size, axis=0)
                rows[:, column] = values
                if column != axis:
                    rows[:, 2 * n_dims + axis] = values + offset
                best = int(np.argmax(scorer(rows)))
                if abs(values[best] - theta[column]) <= 2 * step_m:
                    continue
                candidate, cand_ll, _ = fit_lattice(r, rows[best], max_order=max_order, model=model)
                if cand_ll > loglik + 1.0:
                    theta, loglik, improved = candidate, cand_ll, True
        if not improved:
            break
    return theta, loglik


# --------------------------------------------------------------------- results
def _canonical(
    theta: NDArray[np.float64], n_dims: int
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Sort the axes longest first and pick the canonical per-axis layout."""
    size = theta[:n_dims].copy()
    mic = theta[n_dims : 2 * n_dims].copy()
    src = theta[2 * n_dims : 3 * n_dims].copy()
    for axis in range(n_dims):
        if src[axis] < mic[axis]:
            mic[axis], src[axis] = src[axis], mic[axis]
        if mic[axis] > size[axis] - src[axis]:
            mic[axis], src[axis] = size[axis] - src[axis], size[axis] - mic[axis]
    order = np.argsort(-size, kind="stable")
    out = np.concatenate([size[order], mic[order], src[order], theta[-1:]])
    return out, order


def _fit_quality(
    r: NDArray[np.float64], theta: NDArray[np.float64], lattice: _Lattice, model: EchoModel
) -> tuple[float, float]:
    """Share of echoes explained, and share of strong predicted arrivals never picked."""
    rho, distance = lattice.lengths(theta)
    n_dims = lattice.n_dims
    low, high, _ = _window(r, model)
    inside = (rho > low) & (rho < high)
    tol = max(2 * model.resolution_m, 0.02)
    q0 = float(np.linalg.norm(theta[2 * n_dims : 3 * n_dims] - theta[n_dims : 2 * n_dims]))
    prob = detection_probability(lattice.n_reflections, distance, q0, model)
    strong = np.sort(rho[inside & (prob >= 0.5 * model.max_detection)])
    explained = float(np.mean(_nearest_gap(np.sort(rho[inside]), r) < tol)) if inside.any() else 0.0
    missed = float(np.mean(_nearest_gap(r, strong) >= tol)) if strong.size else 1.0
    return explained, missed


def invert_lattice(
    observed_m: ArrayLike,
    *,
    n_dims: int = 2,
    latency_m: float | None = None,
    model: EchoModel | None = None,
    max_order: int = 4,
    searches: tuple[str, ...] | None = None,
    n_latencies: int = 8,
    bearing_step_deg: float = 1.0,
    n_refine: int | None = None,
    min_explained: float | None = None,
    max_missed: float | None = None,
) -> LatticeEstimate:
    """Fit a rectangular room to every picked echo of one impulse response.

    Parameters
    ----------
    observed_m : array_like
        Picked echoes as path lengths (arrival time times speed of sound);
        the shortest must be the direct sound.
    n_dims : int, optional
        2 (planar room) or 3.
    latency_m : float, optional
        Known playback latency as a path length; ``None`` (default)
        estimates it.
    model : EchoModel, optional
        Detection model; ``resolution_m`` should be one sample at the
        analysis rate.
    max_order : int, optional
        Largest image index per axis in the model (cover the analyzed span:
        4 for a 2-D room, 5 for a 3-D one).
    searches : tuple of str, optional
        Hypothesis generators from :data:`SEARCH_METHODS`; by default both
        in 2-D and ``image_groups`` in 3-D.
    n_latencies : int, optional
        Latency hypotheses tried when ``latency_m`` is ``None``.
    bearing_step_deg : float, optional
        Grid step of the ``axis_scan`` search.
    n_refine : int, optional
        Distinct hypotheses refined by :func:`fit_lattice` (8 in 2-D, 12 in 3-D).
    min_explained, max_missed : float, optional
        Thresholds of :attr:`LatticeEstimate.reliable`; the defaults (0.9
        and 0.2 in 2-D, 0.9 and 0.1 in 3-D) separated right from wrong
        fits on simulated rooms (see the README).

    Returns
    -------
    LatticeEstimate

    Raises
    ------
    InversionError
        When there are too few echoes or no hypothesis fits them.

    Examples
    --------
    >>> lengths, _ = image_path_lengths([6.0, 4.0], [2.2, 1.3], [3.2, 2.3])
    >>> estimate = invert_lattice(lengths[:25] + 2.0)  # 2 m of path = 5.8 ms of latency
    >>> [round(x, 2) for x in estimate.size_m], round(estimate.latency_m, 2), estimate.reliable
    ([6.0, 4.0], 2.0, True)
    >>> [round(x) for x in estimate.bearing_deg]  # the source sits at 45 degrees
    [45]
    """
    if n_dims not in (2, 3):
        raise ValueError(f"n_dims must be 2 or 3, got {n_dims!r}")
    model = model or EchoModel()
    searches = searches or (("image_groups", "axis_scan") if n_dims == 2 else ("image_groups",))
    n_refine = n_refine or (8 if n_dims == 2 else 12)
    min_explained = 0.9 if min_explained is None else min_explained
    max_missed = (0.2 if n_dims == 2 else 0.1) if max_missed is None else max_missed
    for name in searches:
        if name not in SEARCH_METHODS:
            raise ValueError(f"unknown search {name!r}; choose from {list(SEARCH_METHODS)}")
    r = np.sort(np.asarray(observed_m, dtype=np.float64).ravel())
    r = r[np.isfinite(r)]
    if r.size < 4 * n_dims + 2:
        raise InversionError(
            f"{r.size} echoes are too few for a {n_dims}-D room; "
            f"at least {4 * n_dims + 2} are needed"
        )
    lattice = _Lattice(n_dims, max_order)
    coarse = _Scorer(r, lattice, model, max(0.02, model.resolution_m))
    tolerance = max(model.resolution_m, 0.005)
    latencies = (
        [float(latency_m)]
        if latency_m is not None
        else latency_candidates(r, max_candidates=n_latencies)
    )
    if not latencies:
        raise InversionError("no latency hypothesis: the echoes show no combination of two images")
    grid = bearing_grid(n_dims, bearing_step_deg)
    pool: list[tuple[float, NDArray[np.float64]]] = []
    for delta in latencies:
        if "image_groups" in searches:
            pool += _search_image_groups(r, delta, n_dims, coarse, tolerance)
        if "axis_scan" in searches:
            shifts = np.zeros(1) if latency_m is not None else _LATENCY_SHIFTS_M
            pool += _search_axis_scan(r, delta, n_dims, coarse, grid, shifts_m=shifts)
    if not pool:
        raise InversionError("no room hypothesis fits the echoes")
    pool.sort(key=lambda item: -item[0])
    refined: list[tuple[float, NDArray[np.float64]]] = []
    starts: list[NDArray[np.float64]] = []
    for _, start in pool:
        if any(np.max(np.abs(start - s)) < 0.05 for s in starts):
            continue
        starts.append(start)
        if latency_m is not None:
            start = start.copy()
            start[-1] = latency_m
        theta, loglik, _ = fit_lattice(r, start, max_order=max_order, model=model)
        refined.append((loglik, theta))
        if len(starts) == n_refine:
            break
    refined.sort(key=lambda item: -item[0])
    best_theta, best_ll = refined[0][1], refined[0][0]
    for loglik, theta in refined[: 2 if n_dims == 2 else 3]:
        theta, loglik = _rescan_axes(r, theta, loglik, coarse, max_order=max_order, model=model)
        if loglik > best_ll:
            best_theta, best_ll = theta, loglik
    theta, loglik, covariance = fit_lattice(r, best_theta, max_order=max_order, model=model)
    canonical, order = _canonical(theta, n_dims)
    size = canonical[:n_dims]
    mic = canonical[n_dims : 2 * n_dims]
    src = canonical[2 * n_dims : 3 * n_dims]
    side_sd = np.sqrt(np.clip(np.diag(covariance)[:n_dims], 0.0, None))[order]
    offset = src - mic
    distance = float(np.linalg.norm(offset))
    if n_dims == 2:
        bearing = (float(np.degrees(np.arctan2(offset[1], offset[0]))),)
    else:
        bearing = (
            float(np.degrees(np.arctan2(offset[1], offset[0]))),
            float(np.degrees(np.arctan2(offset[2], np.hypot(offset[0], offset[1])))),
        )
    explained, missed = _fit_quality(r, theta, lattice, model)
    estimate = LatticeEstimate(
        size_m=tuple(float(x) for x in size),
        microphone_m=tuple(float(x) for x in mic),
        source_m=tuple(float(x) for x in src),
        latency_m=float(theta[-1]),
        size_sd_m=tuple(float(x) for x in side_sd),
        latency_sd_m=float(np.sqrt(max(covariance[-1, -1], 0.0))),
        source_distance_m=distance,
        bearing_deg=bearing,
        log_likelihood=float(loglik),
        n_echoes=int(r.size),
        explained_fraction=explained,
        missed_fraction=missed,
        reliable=bool(explained >= min_explained and missed <= max_missed),
    )
    _LOGGER.info(
        "lattice fit: sides %s m, latency %.3f m, explained %.2f, missed %.2f",
        ", ".join(f"{x:.3f}" for x in estimate.size_m),
        estimate.latency_m,
        explained,
        missed,
    )
    return estimate
