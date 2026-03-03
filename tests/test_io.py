"""Tests for py_audio.io — WAV read/write utilities."""

import os
import tempfile

import numpy as np
import pytest

from py_audio.io import read_audio, write_audio


class TestReadAudio:
    """Tests for read_audio."""

    def test_reads_wav_file(self):
        """Round-trip: write then read back and verify."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            data = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float64)
            write_audio(path, data, sample_rate=16000)
            sr, result = read_audio(path)
            assert sr == 16000
            assert result.dtype == np.float64
            assert len(result) == 5
        finally:
            os.unlink(path)

    def test_file_not_found(self):
        """Raise an error for a missing file."""
        with pytest.raises(FileNotFoundError):
            read_audio("/nonexistent/path.wav")


class TestWriteAudio:
    """Tests for write_audio."""

    def test_normalizes_to_int16(self):
        """Peak value maps to 32767 in the written file."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            data = np.array([0.0, 2.0, -2.0], dtype=np.float64)
            write_audio(path, data, sample_rate=44100)
            sr, result = read_audio(path)
            assert sr == 44100
            # Peak should map to ±32767
            assert np.max(np.abs(result)) == pytest.approx(32767.0, abs=1)
        finally:
            os.unlink(path)

    def test_all_zeros(self):
        """Writing all-zero data should not error."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            data = np.zeros(100, dtype=np.float64)
            write_audio(path, data, sample_rate=44100)
            sr, result = read_audio(path)
            assert sr == 44100
            assert np.all(result == 0)
        finally:
            os.unlink(path)

    def test_empty_array_raises(self):
        """Writing an empty array should raise ValueError."""
        with pytest.raises(ValueError):
            write_audio("out.wav", np.array([], dtype=np.float64))

    def test_stereo_round_trip(self):
        """Stereo data should survive a write/read round-trip."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            data = np.column_stack([
                np.sin(np.linspace(0, 2 * np.pi, 1000)),
                np.cos(np.linspace(0, 2 * np.pi, 1000)),
            ])
            write_audio(path, data, sample_rate=22050)
            sr, result = read_audio(path)
            assert sr == 22050
            assert result.shape == (1000, 2)
        finally:
            os.unlink(path)
