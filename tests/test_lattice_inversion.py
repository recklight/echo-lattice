"""Lattice inversion: the latency identity, the point-process score, the fit and the full search."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import simulate
from echo_lattice import (
    SEARCH_METHODS,
    EchoModel,
    InversionError,
    bearing_grid,
    detection_probability,
    fit_lattice,
    image_path_lengths,
    invert_lattice,
    latency_candidates,
    point_process_loglik,
    room_size_error,
)

PLANAR = {"size": (6.0, 4.0), "mic": (2.2, 1.3), "src": (3.2, 2.3)}
TALL = {"size": (5.5, 4.2, 2.9), "mic": (1.9, 2.6, 1.3), "src": (3.4, 1.7, 1.6)}


def _lengths(scene, max_order=3):
    lengths, _ = image_path_lengths(scene["size"], scene["mic"], scene["src"], max_order=max_order)
    return lengths


def test_path_lengths_equal_the_image_source_simulation():
    scene = {"room_m": PLANAR["size"], "microphone_m": PLANAR["mic"], "source_m": PLANAR["src"]}
    truth = simulate(scene, max_order=2)
    lengths, bounces = image_path_lengths(PLANAR["size"], PLANAR["mic"], PLANAR["src"], max_order=2)
    assert np.allclose(lengths, np.sort(truth.distances_m))
    assert bounces[0] == 0 and np.all(bounces[1:] >= 1)


def test_images_on_different_axes_give_the_latency_in_closed_form():
    """|A+B|^2 + |0|^2 = |A|^2 + |B|^2, and a common latency drops out of the squares."""
    size, mic, src = np.array(PLANAR["size"]), np.array(PLANAR["mic"]), np.array(PLANAR["src"])
    direct = np.linalg.norm(src - mic)
    wall_x = np.hypot(src[0] + mic[0], src[1] - mic[1])  # image (-1, 0)
    wall_y = np.hypot(src[0] - mic[0], 2 * size[1] - src[1] - mic[1])  # image (0, 1)
    corner = np.hypot(src[0] + mic[0], 2 * size[1] - src[1] - mic[1])  # image (-1, 1)
    assert corner**2 + direct**2 == pytest.approx(wall_x**2 + wall_y**2)
    delta = 1.7
    r0, ra, rb, rab = direct + delta, wall_x + delta, wall_y + delta, corner + delta
    closed_form = (rab**2 + r0**2 - ra**2 - rb**2) / (2 * (rab + r0 - ra - rb))
    assert closed_form == pytest.approx(delta)


def test_the_true_latency_is_the_first_candidate():
    candidates = latency_candidates(_lengths(PLANAR)[:25] + 2.3)
    assert candidates[0] == pytest.approx(2.3, abs=1e-3)
    assert len(candidates) <= 8
    assert all(abs(a - b) > 0.08 for i, a in enumerate(candidates) for b in candidates[:i])


def test_exact_echoes_give_the_room_the_latency_and_the_bearing():
    estimate = invert_lattice(_lengths(PLANAR)[:30] + 1.25)
    assert np.allclose(estimate.size_m, (6.0, 4.0), atol=1e-3)
    assert estimate.latency_m == pytest.approx(1.25, abs=1e-3)
    assert estimate.source_distance_m == pytest.approx(np.sqrt(2.0), abs=1e-3)
    assert estimate.bearing_deg[0] == pytest.approx(45.0, abs=0.1)
    assert estimate.reliable and estimate.explained_fraction > 0.95
    assert max(estimate.size_sd_m) < 0.01 and estimate.height_m is None


def test_the_reported_layout_reproduces_the_echoes():
    """Positions are one of the equivalent layouts: they must give the same path lengths."""
    estimate = invert_lattice(_lengths(PLANAR)[:30] + 0.4)
    refit = _lengths(
        {"size": estimate.size_m, "mic": estimate.microphone_m, "src": estimate.source_m}
    )
    assert np.allclose(refit[:30], _lengths(PLANAR)[:30], atol=2e-3)
    offset = np.subtract(estimate.source_m, estimate.microphone_m)
    assert np.all(offset >= 0)
    assert np.all(
        np.array(estimate.microphone_m) <= np.subtract(estimate.size_m, estimate.source_m) + 1e-9
    )


def test_a_known_latency_is_kept():
    estimate = invert_lattice(_lengths(PLANAR)[:30] + 0.9, latency_m=0.9)
    assert estimate.latency_m == pytest.approx(0.9, abs=2e-3)
    assert room_size_error(estimate.size_m, PLANAR["size"]) < 0.005


def test_missing_jittered_and_spurious_echoes_are_tolerated(rng):
    lengths = _lengths(PLANAR)[:32] + 3.0
    lengths = np.delete(lengths, [4, 9, 15])  # three echoes never picked
    lengths[1:] += rng.normal(0.0, 0.003, lengths.size - 1)  # 3 mm of timing noise
    lengths = np.sort(np.concatenate([lengths, rng.uniform(lengths[1], lengths[-1], 3)]))
    estimate = invert_lattice(lengths, model=EchoModel(resolution_m=343 / 48_000))
    assert room_size_error(estimate.size_m, PLANAR["size"]) < 0.02
    assert estimate.latency_m == pytest.approx(3.0, abs=0.01)
    assert estimate.reliable


def test_a_three_dimensional_room_from_exact_echoes():
    lengths = _lengths(TALL, max_order=5)[:90] + 0.6
    estimate = invert_lattice(
        lengths, n_dims=3, max_order=5, model=EchoModel(resolution_m=343 / 48_000)
    )
    assert room_size_error(estimate.size_m, TALL["size"]) < 0.01
    assert estimate.height_m == pytest.approx(2.9, abs=0.01)
    assert estimate.latency_m == pytest.approx(0.6, abs=0.005)
    assert len(estimate.bearing_deg) == 2 and estimate.reliable


def test_random_echoes_are_not_called_reliable(rng):
    lengths = np.sort(np.concatenate([[1.2], rng.uniform(2.0, 20.0, 30)]))
    try:
        estimate = invert_lattice(lengths, n_latencies=3)
    except InversionError:
        return
    assert not estimate.reliable


def test_too_few_echoes_are_rejected():
    with pytest.raises(InversionError, match="too few"):
        invert_lattice(_lengths(PLANAR)[:6])
    with pytest.raises(ValueError, match="n_dims"):
        invert_lattice(_lengths(PLANAR)[:30], n_dims=4)
    with pytest.raises(ValueError, match="unknown search"):
        invert_lattice(_lengths(PLANAR)[:30], searches=("brute_force",))


def test_the_point_process_prefers_the_room_to_its_half_size_copy():
    """A room with half the sides contains the true image lattice; the missed-echo term must
    still rank it lower."""
    model = EchoModel()
    observed = _lengths(PLANAR, max_order=4)[:30]
    scores = []
    for size, mic, src in [
        (PLANAR["size"], PLANAR["mic"], PLANAR["src"]),
        ((3.0, 2.0), (2.2, 1.3), (2.8, 1.7)),  # every true image is also one of its images
    ]:
        lengths, bounces = image_path_lengths(size, mic, src, max_order=8)
        prob = detection_probability(bounces, lengths, lengths[0], model)
        scores.append(point_process_loglik(observed, lengths, prob, 0.01, model))
    assert scores[0] > scores[1] + 10


def test_the_fit_converges_from_a_perturbed_start(rng):
    lengths = _lengths(TALL, max_order=5)[:80] + 0.3
    truth = np.concatenate([TALL["size"], TALL["mic"], TALL["src"], [0.3]])
    start = truth + rng.uniform(-0.03, 0.03, truth.size)
    model = EchoModel(resolution_m=0.007)
    theta, loglik, cov = fit_lattice(lengths, start, max_order=5, model=model)
    assert np.allclose(theta, truth, atol=1e-3)
    assert np.isfinite(loglik) and cov.shape == (10, 10)
    assert np.all(np.sqrt(np.diag(cov)[:3]) < 0.01)


def test_detection_probability_falls_with_bounces_and_distance():
    model = EchoModel()
    near = detection_probability([1, 1, 3], [3.0, 12.0, 3.0], 1.0, model)
    assert near[0] > near[1] and near[0] > near[2]
    assert np.all((near > 0) & (near <= model.max_detection))


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"relative_floor": 0.0}, "relative_floor"),
        ({"reflection": 1.2}, "reflection"),
        ({"clutter": 0.0}, "clutter"),
        ({"max_detection": 0.0}, "max_detection"),
        ({"resolution_m": 0.0}, "resolution_m"),
    ],
)
def test_invalid_echo_models_are_rejected(kwargs, fragment):
    with pytest.raises(ValueError, match=fragment):
        EchoModel(**kwargs)


def test_the_bearing_grid_covers_the_positive_quadrant_and_octant():
    planar = bearing_grid(2, 10.0)
    assert planar.shape == (10, 2) and np.allclose(np.linalg.norm(planar, axis=1), 1.0)
    spatial = bearing_grid(3, 15.0)
    assert np.all(spatial >= -1e-12) and np.allclose(np.linalg.norm(spatial, axis=1), 1.0)
    assert SEARCH_METHODS == ("image_groups", "axis_scan")


def test_the_estimate_flattens_to_a_table_row():
    estimate = invert_lattice(_lengths(PLANAR)[:30] + 0.5)
    row = estimate.as_row(343.0)
    assert row["latency_ms"] == pytest.approx(1000 * 0.5 / 343.0, abs=0.01)
    assert np.isnan(row["height_m"]) and np.isnan(row["elevation_deg"])
    assert set(estimate.as_dict()) >= {"size_m", "latency_m", "reliable"}
