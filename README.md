# Py_Audio

A modular Python package for audio processing — read, transform, analyse, and
filter WAV files with a simple API built on NumPy and SciPy.

## Features

| Module | Functions | Description |
|--------|-----------|-------------|
| `py_audio.io` | `read_audio`, `write_audio` | WAV file I/O with float64 conversion |
| `py_audio.effects` | `reverse_audio`, `change_speed`, `change_volume` | Time/amplitude effects |
| `py_audio.analysis` | `compute_fft`, `spectrum_analysis` | FFT spectrum analysis |
| `py_audio.filters` | `bandpass_filter` | Butterworth bandpass filter |

## Quick Start

```bash
pip install -r requirements.txt
```

```python
from py_audio import read_audio, write_audio, reverse_audio
from py_audio import spectrum_analysis, bandpass_filter

# Read a WAV file
sample_rate, audio = read_audio("data/Faded.wav")

# Reverse the audio
reversed_audio = reverse_audio(audio)
write_audio("data/rev_audio.wav", reversed_audio, sample_rate)

# Bandpass filter (200 Hz – 4000 Hz)
filtered = bandpass_filter(audio, sample_rate, low_freq=200, high_freq=4000)
write_audio("data/filtered.wav", filtered, sample_rate)

# Spectrum analysis
info = spectrum_analysis("data/Faded.wav")
print(f"Peak frequency: {info['peak_frequency']:.1f} Hz")
```

See `example.py` for a full runnable demo.

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

## Project Structure

```
py_audio/          # Core package
  __init__.py      #   Public API
  io.py            #   WAV read/write
  effects.py       #   Audio effects
  analysis.py      #   FFT & spectrum analysis
  filters.py       #   Digital filters
tests/             # Unit tests
example.py         # Demo script
requirements.txt   # Dependencies
```
