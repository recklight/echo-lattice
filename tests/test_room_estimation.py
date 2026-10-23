"""The estimation chain: simulated scene, identification, inversion."""

from __future__ import annotations

import numpy as np
import pytest

from echo_lattice import (
    Config,
    ConfigError,
    InversionError,
    LatticeEstimate,
    delay_response,
    invert_response,
    latency_taps,
    simulate_scene,
)
from echo_lattice.room_estimation import identify_recording, simulate_truth


@pytest.fixture
def config():
    return Config().override({"synthetic.duration_s": 1.0})


def test_a_noisy_white_noise_measurement_gives_the_room_within_centimeters(config):
    scene = simulate_scene(config, rng=0)
    assert scene.misalignment_db < -25.0
    assert scene.room_error_m < 0.02
    assert scene.tap_shift == pytest.approx(0.0)
    assert scene.inversion.estimate.source_distance_m == pytest.approx(np.sqrt(2.0), abs=0.01)
    assert scene.learning_curve().shape == scene.recording.excitation.shape


def test_the_scene_is_reproducible(config):
    first = simulate_scene(config, rng=4)
    second = simulate_scene(config, rng=4)
    assert np.array_equal(first.weights, second.weights)
    assert first.inversion.candidates.equals(second.inversion.candidates)


def test_least_squares_identification_runs_the_same_chain(config):
    scene = simulate_scene(config.override({"identification.method": "least_squares"}), rng=0)
    assert scene.error.size == 0 and scene.learning_curve().size == 0
    assert scene.room_error_m < 0.02


def test_a_given_excitation_and_a_noise_free_recording(config, rng):
    excitation = rng.standard_normal(12_000)
    scene = simulate_scene(config, excitation=excitation, snr_db=None)
    assert np.array_equal(scene.recording.excitation, excitation)
    assert scene.misalignment_db < -40.0


def test_identify_recording_uses_the_configured_method(config):
    scene = simulate_scene(config, rng=1)
    weights = identify_recording(scene.recording, config)
    assert weights.shape == (config.identification.n_taps,)
    assert int(np.argmax(np.abs(weights))) == 66


def test_truth_follows_the_room_section(config):
    moved = config.override({"room.source_m": [6.35, 3.75]})
    assert simulate_truth(moved).taps[0] == round(16_000 / 343.0)


def test_an_empty_response_cannot_be_inverted(config):
    response = np.zeros(1600)
    response[100] = 1.0
    with pytest.raises(InversionError):
        invert_response(response, config)


#: A planar scene measured with every image order, a delayed microphone channel and the
#: source 17 degrees off the long side.
LATTICE_SCENE = {
    "synthetic.duration_s": 1.0,
    "synthetic.latency_ms": 4.0,
    "acoustics.max_order": 4,
    "acoustics.reflection": 0.85,
    "acoustics.delay": "fractional",
    "room.source_m": [7.1, 4.3],
    "identification.method": "least_squares",
    "inversion.method": "lattice",
}


def test_the_lattice_method_recovers_room_latency_and_bearing():
    scene = simulate_scene(Config().override(LATTICE_SCENE), rng=0)
    estimate = scene.inversion.estimate
    assert isinstance(estimate, LatticeEstimate)
    assert scene.room_error_m < 0.01
    assert estimate.latency_ms(343.0) == pytest.approx(4.0, abs=0.02)
    assert estimate.bearing_deg[0] == pytest.approx(np.degrees(np.arctan2(0.55, 1.75)), abs=0.5)
    assert estimate.reliable
    assert list(scene.inversion.candidates.columns)[:3] == ["width_m", "depth_m", "height_m"]


def test_the_first_order_method_mistakes_the_latency_for_distance():
    config = Config().override({**LATTICE_SCENE, "inversion.method": "first_order"})
    scene = simulate_scene(config, rng=0)
    assert scene.room_error_m > 1.0  # 4 ms of latency is 1.37 m of extra path on every echo


def test_latency_delays_the_microphone_channel():
    config = Config().override({"synthetic.latency_ms": 2.5})
    assert latency_taps(config) == 40
    assert delay_response([1.0, 2.0, 3.0], 0).tolist() == [1.0, 2.0, 3.0]
    assert delay_response([1.0, 2.0, 3.0], 2).tolist() == [0.0, 0.0, 1.0]


def test_the_lattice_dimensions_must_match_the_room():
    config = Config().override({"inversion.method": "lattice", "lattice.dimensions": 3})
    with pytest.raises(ConfigError, match=r"lattice\.dimensions=3"):
        simulate_scene(config, rng=0)
