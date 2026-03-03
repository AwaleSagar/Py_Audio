"""Audio effects: reverse, speed change, and volume adjustment."""

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample


def reverse_audio(data: NDArray[np.float64]) -> NDArray[np.float64]:
    """Reverse an audio array along the time axis.

    Works for both mono (1-D) and stereo (2-D, shape ``(samples, channels)``)
    arrays.

    Args:
        data: Audio samples as a numpy array.

    Returns:
        A new array with samples in reverse order.
    """
    return np.flip(data, axis=0)


def change_speed(
    data: NDArray[np.float64],
    factor: float,
) -> NDArray[np.float64]:
    """Change playback speed by resampling.

    A *factor* greater than 1 speeds up the audio (fewer output samples);
    a *factor* less than 1 slows it down (more output samples).

    .. note::
       Very large factors may reduce the audio to just a few samples,
       which is unlikely to be musically useful.

    Args:
        data: Audio samples as a numpy array (1-D or 2-D).
        factor: Speed multiplier.  Must be positive.

    Returns:
        Resampled audio array.

    Raises:
        ValueError: If *factor* is not positive.
    """
    if factor <= 0:
        raise ValueError("Speed factor must be positive.")

    num_samples = data.shape[0]
    new_length = int(round(num_samples / factor))
    if new_length == 0:
        new_length = 1

    return resample(data, new_length, axis=0)


def change_volume(
    data: NDArray[np.float64],
    gain_db: float,
) -> NDArray[np.float64]:
    """Apply a gain adjustment in decibels.

    Args:
        data: Audio samples as a numpy array.
        gain_db: Gain to apply in dB.  Positive values amplify,
                 negative values attenuate.

    Returns:
        Gain-adjusted audio array.
    """
    multiplier = 10.0 ** (gain_db / 20.0)
    return data * multiplier
