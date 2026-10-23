"""Room estimation end to end: recording, identified response, echoes, room.

The chain behind the ``simulate``, ``invert`` and ``sensitivity`` commands:

1. :func:`identify_recording`: an adaptive filter turns the excitation and
   the microphone channel into an impulse response
   (:mod:`echo_lattice.identification`);
2. :func:`invert_response`: the response is pruned and its echoes are
   picked and converted to distances; with ``inversion.method =
   first_order`` the nine strongest are labeled in closed form and the
   candidates ranked by resynthesis
   (:mod:`echo_lattice.inversion`), with ``lattice`` every echo
   from the direct sound on is fitted with the image lattice, latency and
   bearing included (:mod:`echo_lattice.lattice_inversion`);
3. :func:`simulate_scene`: the same chain on a recording simulated from the
   ``room`` scene with the image-source model, with the truth kept for
   comparison.

Every setting comes from one :class:`~echo_lattice.config.Config`,
so the sampling rate, the speed of sound and the image-source model are the
same in simulation, distance conversion and resynthesis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from .echoes import pick_arrivals, pick_echoes, tap_shift_error, taps_to_distances
from .errors import ConfigError
from .excitation import make_excitation
from .geometry import room_size_error
from .identification import (
    identify_response,
    learning_curve_db,
    nlms,
    normalized_misalignment_db,
    prune_small_taps,
)
from .image_source import ImageSourceResult, image_source_rir
from .inversion import RoomEstimate, best_estimate, invert_echoes, rank_candidates
from .lattice_inversion import EchoModel, LatticeEstimate, invert_lattice
from .recording import Recording, simulate_recording

if TYPE_CHECKING:
    from .config import Config

_LOGGER = logging.getLogger(__name__)

#: Window of the learning curve, in samples.
LEARNING_CURVE_WINDOW: int = 80


@dataclass(frozen=True, eq=False)
class InversionResult:
    """Everything the inversion of one response produced.

    Attributes
    ----------
    response : numpy.ndarray, shape (n_taps,)
        The pruned, peak-normalized magnitude that was analyzed.
    echo_taps : numpy.ndarray
        Picked arrivals in samples, increasing.
    distances_m : numpy.ndarray
        The arrivals as travel distances.
    candidates : pandas.DataFrame
        Ranked candidate layouts (see
        :func:`~echo_lattice.inversion.rank_candidates`); for the
        lattice inversion, one row with the fitted room.
    estimate : RoomEstimate or LatticeEstimate
        The best candidate.

    Examples
    --------
    >>> from echo_lattice.config import Config
    >>> truth = image_source_rir([6.0, 4.0], [3.2, 2.3], [2.2, 1.3], sample_rate=16_000)
    >>> result = invert_response(truth.rir, Config())
    >>> result.echo_taps.shape == result.distances_m.shape, len(result.candidates) > 0
    (True, True)
    """

    response: NDArray[np.float64]
    echo_taps: NDArray[np.float64]
    distances_m: NDArray[np.float64]
    candidates: pd.DataFrame
    estimate: RoomEstimate | LatticeEstimate


@dataclass(frozen=True, eq=False)
class SceneResult:
    """A simulated measurement of the ``room`` scene and its analysis.

    Attributes
    ----------
    truth : ImageSourceResult
        The simulated impulse response and its image sources.
    recording : Recording
        Excitation and noisy microphone signal.
    weights : numpy.ndarray
        Identified response (raw filter weights).
    error : numpy.ndarray
        A-priori error of the adaptive filter (empty for least squares).
    misalignment_db : float
        Normalized misalignment between the identified and the true response.
    inversion : InversionResult
        Echo inversion of the identified response.
    room_error_m : float
        Sorted-side size error of the estimate.
    tap_shift : float
        Mean shift of the picked echoes from the true arrivals, in samples.

    Examples
    --------
    >>> from echo_lattice.config import Config
    >>> scene = simulate_scene(Config().override({"synthetic.duration_s": 0.5}), rng=0)
    >>> scene.learning_curve().shape, scene.recording.sample_rate
    ((8000,), 16000)
    """

    truth: ImageSourceResult
    recording: Recording
    weights: NDArray[np.float64]
    error: NDArray[np.float64]
    misalignment_db: float
    inversion: InversionResult
    room_error_m: float
    tap_shift: float

    def learning_curve(self) -> NDArray[np.float64]:
        """Squared error over the last :data:`LEARNING_CURVE_WINDOW` samples, in dB."""
        if self.error.size == 0:
            return np.zeros(0)
        return learning_curve_db(self.error, LEARNING_CURVE_WINDOW)


def identify_recording(recording: Recording, config: Config) -> NDArray[np.float64]:
    """Identified impulse response of a measurement, with the configured method.

    Parameters
    ----------
    recording : Recording
        Excitation and microphone, at ``acoustics.sample_rate``.
    config : Config
        ``identification`` settings are used.

    Returns
    -------
    numpy.ndarray, shape (identification.n_taps,)
        Raw filter weights.

    Examples
    --------
    >>> from echo_lattice.config import Config
    >>> scene = simulate_scene(Config().override({"synthetic.duration_s": 0.5}), rng=0)
    >>> int(np.argmax(np.abs(identify_recording(scene.recording, Config()))))
    66
    """
    settings = config.identification
    return identify_response(
        recording.excitation,
        recording.microphone,
        method=settings.method,
        n_taps=settings.n_taps,
        step=settings.step,
        regularization=settings.regularization,
    )


def invert_response(response: ArrayLike, config: Config) -> InversionResult:
    """Pick the echoes of a response and estimate the room from them.

    Parameters
    ----------
    response : array_like, shape (n_taps,)
        Identified or simulated impulse response.
    config : Config
        ``identification.prune_threshold``, ``echoes``, ``inversion``,
        ``lattice`` and ``acoustics`` are used.

    Returns
    -------
    InversionResult

    Raises
    ------
    InversionError
        When too few echoes are found or no layout fits them.

    Examples
    --------
    >>> from echo_lattice.config import Config
    >>> config = Config()
    >>> truth = image_source_rir([6.0, 4.0], [3.2, 2.3], [2.2, 1.3], sample_rate=16_000)
    >>> result = invert_response(truth.rir, config)
    >>> [round(side, 2) for side in result.estimate.size_m], result.echo_taps.size
    ([6.0, 4.0], 9)
    """
    if config.inversion.method == "lattice":
        return _invert_lattice_response(response, config)
    acoustics = config.acoustics
    magnitude = prune_small_taps(response, config.identification.prune_threshold)
    taps = pick_echoes(magnitude, **config.echoes.options())
    distances = taps_to_distances(taps, acoustics.sample_rate, acoustics.speed)
    candidates = invert_echoes(
        distances,
        bearing_deg=config.inversion.source_bearing_deg,
        refine=config.inversion.refine,
        max_candidates=config.inversion.max_candidates,
    )
    ranked = rank_candidates(
        candidates,
        magnitude,
        sample_rate=acoustics.sample_rate,
        speed_of_sound_m_s=acoustics.speed,
        metric=config.inversion.rank_metric,
        tolerance_taps=config.inversion.tolerance_taps,
        echo_taps=taps,
        max_order=acoustics.max_order,
        order_mode=acoustics.order_mode,
        reflection=acoustics.reflection,
        delay=acoustics.delay,
    )
    return InversionResult(
        response=magnitude,
        echo_taps=taps,
        distances_m=distances,
        candidates=ranked,
        estimate=best_estimate(ranked),
    )


def echo_model(config: Config) -> EchoModel:
    """Detection model of the lattice inversion from the ``lattice`` and ``acoustics`` settings.

    Examples
    --------
    >>> from echo_lattice.config import Config
    >>> model = echo_model(Config())
    >>> model.relative_floor, round(model.resolution_m, 5)
    (0.02, 0.02144)
    """
    settings, acoustics = config.lattice, config.acoustics
    return EchoModel(
        relative_floor=settings.relative_floor,
        reflection=settings.reflection,
        clutter=settings.clutter,
        resolution_m=acoustics.speed / acoustics.sample_rate,
    )


def _invert_lattice_response(response: ArrayLike, config: Config) -> InversionResult:
    acoustics, settings = config.acoustics, config.lattice
    magnitude = prune_small_taps(response, config.identification.prune_threshold)
    taps = pick_arrivals(
        magnitude,
        settings.max_echoes,
        relative_floor=settings.relative_floor,
        min_separation_taps=settings.min_separation_taps,
    )
    distances = taps_to_distances(taps, acoustics.sample_rate, acoustics.speed)
    latency = None if settings.latency_ms is None else 1e-3 * settings.latency_ms * acoustics.speed
    estimate = invert_lattice(
        distances,
        n_dims=settings.dimensions,
        latency_m=latency,
        model=echo_model(config),
        max_order=settings.max_order,
        searches=settings.searches,
        n_latencies=settings.n_latencies,
        bearing_step_deg=settings.bearing_step_deg,
        min_explained=settings.min_explained,
        max_missed=settings.max_missed,
    )
    return InversionResult(
        response=magnitude,
        echo_taps=taps,
        distances_m=distances,
        candidates=pd.DataFrame([estimate.as_row(acoustics.speed)]),
        estimate=estimate,
    )


def latency_taps(config: Config) -> int:
    """Whole-sample delay of the microphone channel set by ``synthetic.latency_ms``.

    Examples
    --------
    >>> from echo_lattice.config import Config
    >>> latency_taps(Config().override({"synthetic.latency_ms": 5.0}))
    80
    """
    return round(1e-3 * config.synthetic.latency_ms * config.acoustics.sample_rate)


def delay_response(response: ArrayLike, taps: int) -> NDArray[np.float64]:
    """The response shifted ``taps`` samples later, same length.

    Examples
    --------
    >>> delay_response([1.0, 0.5, 0.25, 0.0], 1).tolist()
    [0.0, 1.0, 0.5, 0.25]
    """
    values = np.asarray(response, dtype=np.float64).ravel()
    if taps <= 0:
        return values.copy()
    return np.concatenate([np.zeros(taps), values])[: values.size]


def simulate_truth(
    config: Config,
    *,
    room_m: ArrayLike | None = None,
    microphone_m: ArrayLike | None = None,
    source_m: ArrayLike | None = None,
    n_taps: int | None = None,
) -> ImageSourceResult:
    """Image-source response of the ``room`` scene (or of the given positions).

    Examples
    --------
    >>> from echo_lattice.config import Config
    >>> simulate_truth(Config()).arrival_taps().tolist()
    [66, 307, 399, 455, 545, 548, 602, 624, 675]
    """
    acoustics, room = config.acoustics, config.room
    return image_source_rir(
        room.size_m if room_m is None else room_m,
        room.source_m if source_m is None else source_m,
        room.microphone_m if microphone_m is None else microphone_m,
        sample_rate=acoustics.sample_rate,
        n_taps=acoustics.rir_taps if n_taps is None else int(n_taps),
        speed_of_sound_m_s=acoustics.speed,
        max_order=acoustics.max_order,
        order_mode=acoustics.order_mode,
        reflection=acoustics.reflection,
        delay=acoustics.delay,
    )


def simulate_scene(
    config: Config,
    *,
    rng: np.random.Generator | int | None = None,
    excitation: ArrayLike | None = None,
    snr_db: float | str | None = "config",
    truth: ImageSourceResult | None = None,
) -> SceneResult:
    """Simulate a measurement of the ``room`` scene and run the whole estimation chain.

    Parameters
    ----------
    config : Config
        Complete configuration: ``room`` and ``acoustics`` define the scene,
        ``synthetic`` the excitation, its length and the noise.
    rng : numpy.random.Generator or int, optional
        Random generator or seed of excitation and noise.
    excitation : array_like, optional
        A signal to play instead of the synthetic ``synthetic.excitation``.
    snr_db : float, None or "config", optional
        Signal-to-noise ratio of the recording; ``None`` adds no noise and
        ``"config"`` (the default) uses ``synthetic.snr_db``.
    truth : ImageSourceResult, optional
        A precomputed true response (defaults to :func:`simulate_truth`).

    Returns
    -------
    SceneResult

    Examples
    --------
    >>> from echo_lattice.config import Config
    >>> scene = simulate_scene(Config().override({"synthetic.duration_s": 0.5}), rng=0)
    >>> [round(side, 1) for side in scene.inversion.estimate.size_m], scene.room_error_m < 0.05
    ([10.7, 7.5], True)
    """
    rng = np.random.default_rng(rng)
    acoustics, synthetic = config.acoustics, config.synthetic
    if (
        config.inversion.method == "lattice"
        and len(config.room.size_m) != config.lattice.dimensions
    ):
        raise ConfigError(
            f"room.size_m has {len(config.room.size_m)} sides but "
            f"lattice.dimensions={config.lattice.dimensions}"
        )
    truth = simulate_truth(config) if truth is None else truth
    shift = latency_taps(config)
    delayed = delay_response(truth.rir, shift)
    rate = acoustics.sample_rate
    if excitation is None:
        n_samples = round(synthetic.duration_s * rate)
        signal = make_excitation(synthetic.excitation, n_samples, rate, rng=rng)
    else:
        signal = np.asarray(excitation, dtype=np.float64).ravel()
    level = synthetic.snr_db if snr_db == "config" else snr_db
    microphone = simulate_recording(
        signal,
        delayed,
        snr_db=None if level is None else float(level),
        noise_kind=synthetic.noise,
        sample_rate=rate,
        rng=rng,
    )
    recording = Recording(excitation=signal, microphone=microphone, sample_rate=rate)
    settings = config.identification
    if settings.method == "nlms":
        adaptive = nlms(signal, microphone, settings.n_taps, settings.step, settings.regularization)
        weights, error = adaptive.weights, adaptive.error
    else:
        weights, error = identify_recording(recording, config), np.zeros(0)
    misalignment = normalized_misalignment_db(delayed, weights)
    inversion = invert_response(weights, config)
    room_error = room_size_error(inversion.estimate.size_m, config.room.size_m)
    tap_shift = tap_shift_error(inversion.echo_taps, truth.arrival_taps() + shift)
    _LOGGER.info(
        "misalignment %.1f dB, estimated room %.3f x %.3f m (size error %.3f m)",
        misalignment,
        inversion.estimate.width_m,
        inversion.estimate.depth_m,
        room_error,
    )
    return SceneResult(
        truth=truth,
        recording=recording,
        weights=weights,
        error=error,
        misalignment_db=misalignment,
        inversion=inversion,
        room_error_m=room_error,
        tap_shift=tap_shift,
    )
