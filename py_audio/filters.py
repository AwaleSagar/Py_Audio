"""Digital audio filters built on scipy's signal processing toolkit."""

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfilt


def bandpass_filter(
    data: NDArray[np.float64],
    sample_rate: int,
    low_freq: float,
    high_freq: float,
    order: int = 5,
) -> NDArray[np.float64]:
    """Apply a Butterworth bandpass filter to audio data.

    Uses second-order sections (``sos``) for numerical stability.
    Works with both mono (1-D) and stereo (2-D) arrays.  For stereo
    input each channel is filtered independently.

    Args:
        data: Audio samples as a numpy array.
        sample_rate: Sample rate in Hz.
        low_freq: Lower cutoff frequency in Hz.
        high_freq: Upper cutoff frequency in Hz.
        order: Filter order (default 5).

    Returns:
        Filtered audio as a numpy array with the same shape as *data*.

    Raises:
        ValueError: If the frequency parameters are invalid.
    """
    nyquist = sample_rate / 2.0

    if low_freq <= 0 or high_freq <= 0:
        raise ValueError("Cutoff frequencies must be positive.")
    if low_freq >= high_freq:
        raise ValueError("low_freq must be less than high_freq.")
    if high_freq >= nyquist:
        raise ValueError(
            f"high_freq ({high_freq} Hz) must be below the Nyquist "
            f"frequency ({nyquist} Hz)."
        )

    low = low_freq / nyquist
    high = high_freq / nyquist

    sos = butter(order, [low, high], btype="band", output="sos")
    return sosfilt(sos, data, axis=0)
