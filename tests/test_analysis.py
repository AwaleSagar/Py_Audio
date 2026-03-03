"""Tests for py_audio.analysis — FFT and spectrum analysis."""

import os
import tempfile

import numpy as np
import pytest

from py_audio.analysis import compute_fft, spectrum_analysis
from py_audio.io import write_audio


class TestComputeFFT:
    """Tests for compute_fft."""

    def test_known_sine(self):
        """A pure sine wave should have its peak at the expected frequency."""
        sr = 8000
        freq = 440.0
        t = np.arange(0, 1.0, 1.0 / sr)
        data = np.sin(2 * np.pi * freq * t)

        frequencies, magnitudes = compute_fft(data, sr)
        peak_idx = np.argmax(magnitudes[1:]) + 1  # skip DC
        detected_freq = frequencies[peak_idx]

        assert detected_freq == pytest.approx(freq, abs=sr / len(data))

    def test_stereo_input(self):
        """Stereo input should be mixed to mono before FFT."""
        sr = 8000
        t = np.arange(0, 0.5, 1.0 / sr)
        mono = np.sin(2 * np.pi * 200 * t)
        stereo = np.column_stack([mono, mono])

        freqs_m, mags_m = compute_fft(mono, sr)
        freqs_s, mags_s = compute_fft(stereo, sr)

        np.testing.assert_array_almost_equal(freqs_m, freqs_s)
        np.testing.assert_array_almost_equal(mags_m, mags_s)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            compute_fft(np.array([]), 44100)


class TestSpectrumAnalysis:
    """Tests for spectrum_analysis."""

    def test_returns_expected_keys(self):
        """Result dict should have all expected keys."""
        sr = 16000
        t = np.arange(0, 0.5, 1.0 / sr)
        data = np.sin(2 * np.pi * 1000 * t)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            write_audio(path, data, sample_rate=sr)
            result = spectrum_analysis(path)
            assert set(result.keys()) == {
                "frequencies",
                "magnitudes",
                "sample_rate",
                "duration",
                "peak_frequency",
            }
            assert result["sample_rate"] == sr
            assert result["duration"] == pytest.approx(0.5, abs=1.0 / sr)
        finally:
            os.unlink(path)
