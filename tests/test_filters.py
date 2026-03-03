"""Tests for py_audio.filters — digital audio filters."""

import numpy as np
import pytest

from py_audio.filters import bandpass_filter


class TestBandpassFilter:
    """Tests for bandpass_filter."""

    def test_passes_in_band_signal(self):
        """A sine wave within the passband should survive the filter."""
        sr = 8000
        t = np.arange(0, 1.0, 1.0 / sr)
        data = np.sin(2 * np.pi * 500 * t)  # 500 Hz in band

        filtered = bandpass_filter(data, sr, low_freq=200, high_freq=1000)
        # Energy should be mostly preserved (> 50% of original)
        assert np.std(filtered) > 0.5 * np.std(data)

    def test_attenuates_out_of_band(self):
        """A sine wave outside the passband should be attenuated."""
        sr = 8000
        t = np.arange(0, 1.0, 1.0 / sr)
        data = np.sin(2 * np.pi * 100 * t)  # 100 Hz, below passband

        filtered = bandpass_filter(data, sr, low_freq=500, high_freq=2000)
        # Energy should be significantly reduced
        assert np.std(filtered) < 0.3 * np.std(data)

    def test_invalid_freq_order(self):
        with pytest.raises(ValueError, match="less than"):
            bandpass_filter(np.ones(100), 8000, low_freq=2000, high_freq=500)

    def test_freq_above_nyquist(self):
        with pytest.raises(ValueError, match="Nyquist"):
            bandpass_filter(np.ones(100), 8000, low_freq=100, high_freq=5000)

    def test_negative_freq(self):
        with pytest.raises(ValueError, match="positive"):
            bandpass_filter(np.ones(100), 8000, low_freq=-10, high_freq=500)

    def test_stereo_input(self):
        """Stereo array should retain its shape after filtering."""
        sr = 8000
        t = np.arange(0, 0.5, 1.0 / sr)
        stereo = np.column_stack([
            np.sin(2 * np.pi * 300 * t),
            np.cos(2 * np.pi * 300 * t),
        ])
        filtered = bandpass_filter(stereo, sr, low_freq=100, high_freq=1000)
        assert filtered.shape == stereo.shape
