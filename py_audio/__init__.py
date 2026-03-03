"""py_audio – A modular Python package for audio processing.

Provides WAV I/O, effects, spectrum analysis, and digital filtering
built on NumPy and SciPy.

Quick start::

    from py_audio import read_audio, write_audio, reverse_audio
    from py_audio import spectrum_analysis, bandpass_filter
"""

from py_audio.analysis import compute_fft, spectrum_analysis
from py_audio.effects import change_speed, change_volume, reverse_audio
from py_audio.filters import bandpass_filter
from py_audio.io import read_audio, write_audio

__all__ = [
    "read_audio",
    "write_audio",
    "reverse_audio",
    "change_speed",
    "change_volume",
    "compute_fft",
    "spectrum_analysis",
    "bandpass_filter",
]
