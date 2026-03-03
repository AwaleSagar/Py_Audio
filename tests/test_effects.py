"""Tests for py_audio.effects — audio effects."""

import numpy as np
import pytest

from py_audio.effects import change_speed, change_volume, reverse_audio


class TestReverseAudio:
    """Tests for reverse_audio."""

    def test_reverses_mono(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = reverse_audio(data)
        np.testing.assert_array_equal(result, [5.0, 4.0, 3.0, 2.0, 1.0])

    def test_reverses_stereo(self):
        data = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        result = reverse_audio(data)
        expected = np.array([[3.0, 30.0], [2.0, 20.0], [1.0, 10.0]])
        np.testing.assert_array_equal(result, expected)


class TestChangeSpeed:
    """Tests for change_speed."""

    def test_speed_up(self):
        data = np.ones(1000)
        result = change_speed(data, factor=2.0)
        assert len(result) == 500

    def test_slow_down(self):
        data = np.ones(1000)
        result = change_speed(data, factor=0.5)
        assert len(result) == 2000

    def test_invalid_factor_raises(self):
        with pytest.raises(ValueError, match="positive"):
            change_speed(np.ones(10), factor=-1.0)

    def test_zero_factor_raises(self):
        with pytest.raises(ValueError, match="positive"):
            change_speed(np.ones(10), factor=0.0)


class TestChangeVolume:
    """Tests for change_volume."""

    def test_gain_positive(self):
        data = np.array([1.0, -1.0])
        result = change_volume(data, gain_db=6.0)
        # +6 dB ≈ 2× amplitude
        assert result[0] == pytest.approx(10 ** (6.0 / 20.0), rel=1e-6)

    def test_gain_zero(self):
        data = np.array([1.0, 0.5, -0.5])
        result = change_volume(data, gain_db=0.0)
        np.testing.assert_array_almost_equal(result, data)

    def test_gain_negative(self):
        data = np.array([1.0])
        result = change_volume(data, gain_db=-20.0)
        assert result[0] == pytest.approx(0.1, rel=1e-6)
