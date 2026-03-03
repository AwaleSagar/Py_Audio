"""Example usage of the py_audio package.

Demonstrates reading a WAV file, applying effects and filters,
running spectrum analysis, and writing results back to disk.
"""

from py_audio import (
    bandpass_filter,
    read_audio,
    reverse_audio,
    spectrum_analysis,
    write_audio,
)


def main() -> None:
    source = "./data/Faded.wav"

    # --- Read the original audio ---
    sample_rate, audio = read_audio(source)
    print(f"Loaded: {source}")
    print(f"  Sample rate : {sample_rate} Hz")
    print(f"  Samples     : {audio.shape[0]}")
    print(f"  Duration    : {audio.shape[0] / sample_rate:.2f} s")
    print()

    # --- Reverse the audio ---
    reversed_audio = reverse_audio(audio)
    write_audio("./data/rev_audio.wav", reversed_audio, sample_rate)
    print("Saved reversed audio  -> data/rev_audio.wav")

    # --- Apply a bandpass filter (200 Hz – 4000 Hz) ---
    filtered = bandpass_filter(audio, sample_rate, low_freq=200, high_freq=4000)
    write_audio("./data/filtered.wav", filtered, sample_rate)
    print("Saved filtered audio  -> data/filtered.wav")

    # --- Spectrum analysis ---
    info = spectrum_analysis(source)
    print()
    print("Spectrum analysis:")
    print(f"  Peak frequency : {info['peak_frequency']:.1f} Hz")
    print(f"  Duration       : {info['duration']:.2f} s")


if __name__ == "__main__":
    main()
