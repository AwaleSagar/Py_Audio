"""Spectrum analysis utilities for audio signals."""

from typing import Any, Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from py_audio.io import read_audio


def compute_fft(
    data: NDArray[np.float64],
    sample_rate: int,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute the single-sided FFT magnitude spectrum of an audio signal.

    For stereo (2-D) input the channels are averaged to mono before the
    FFT is computed.

    Args:
        data: Audio samples (1-D mono or 2-D stereo).
        sample_rate: Sample rate in Hz.

    Returns:
        A tuple ``(frequencies, magnitudes)`` where both are 1-D numpy
        arrays covering frequencies from 0 Hz up to the Nyquist frequency.

    Raises:
        ValueError: If *data* is empty.
    """
    if data.size == 0:
        raise ValueError("Cannot compute FFT of an empty array.")

    # Mix to mono if stereo.
    if data.ndim == 2:
        data = data.mean(axis=1)

    n = len(data)
    fft_vals = np.fft.rfft(data)
    magnitudes = np.abs(fft_vals) * (2.0 / n)
    frequencies = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    return frequencies, magnitudes


def spectrum_analysis(path: str) -> Dict[str, Any]:
    """Read a WAV file and return its spectral characteristics.

    Args:
        path: Path to a WAV file.

    Returns:
        A dictionary with the following keys:

        * ``frequencies`` – 1-D array of FFT bin frequencies (Hz).
        * ``magnitudes``  – 1-D array of magnitude values.
        * ``sample_rate`` – Sample rate of the file (Hz).
        * ``duration``    – Duration of the audio in seconds.
        * ``peak_frequency`` – Frequency (Hz) with the highest magnitude,
          excluding the DC component at index 0.
    """
    sample_rate, data = read_audio(path)

    num_samples = data.shape[0]
    duration = num_samples / sample_rate

    frequencies, magnitudes = compute_fft(data, sample_rate)

    # Find the peak frequency, excluding DC (index 0).
    if len(magnitudes) > 1:
        peak_index = np.argmax(magnitudes[1:]) + 1
        peak_frequency = float(frequencies[peak_index])
    else:
        peak_frequency = 0.0

    return {
        "frequencies": frequencies,
        "magnitudes": magnitudes,
        "sample_rate": sample_rate,
        "duration": duration,
        "peak_frequency": peak_frequency,
    }
