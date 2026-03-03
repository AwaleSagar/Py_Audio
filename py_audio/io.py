"""Audio I/O utilities for reading and writing WAV files."""

from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.io.wavfile import read, write


def read_audio(path: str) -> Tuple[int, NDArray[np.float64]]:
    """Read a WAV file and return its contents as a float64 numpy array.

    Args:
        path: Path to the WAV file.

    Returns:
        A tuple of (sample_rate, data) where data is a float64 numpy array.
        Mono audio is returned as a 1-D array; stereo as a 2-D array with
        shape (num_samples, num_channels).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a valid WAV file.
    """
    sample_rate, data = read(path)
    return sample_rate, data.astype(np.float64, copy=False)


def write_audio(
    path: str,
    data: NDArray[np.float64],
    sample_rate: int = 44100,
) -> None:
    """Normalize audio data to int16 range and write it to a WAV file.

    The data is scaled so that its peak absolute value maps to 32767.
    If the data is all zeros, zeros are written directly.

    Args:
        path: Destination file path (should end with ``.wav``).
        data: Audio samples as a numpy array (1-D mono or 2-D stereo).
        sample_rate: Sample rate in Hz (default 44100).

    Raises:
        ValueError: If *data* is empty.
    """
    if data.size == 0:
        raise ValueError("Cannot write an empty audio array.")

    peak = np.max(np.abs(data))
    if peak == 0:
        scaled = np.zeros_like(data, dtype=np.int16)
    else:
        scaled = np.int16(data / peak * 32767)

    write(path, sample_rate, scaled)
