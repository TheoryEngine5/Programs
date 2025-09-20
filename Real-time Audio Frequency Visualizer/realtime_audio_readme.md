# Real-Time Audio Spectrum Visualizer

**Author:** Emilio Zangger  
**Date:** 2025

---

## Overview

This Python program captures live audio from a microphone and visualizes its frequency spectrum in real time. Using PyAudio for audio input, NumPy for signal processing, and Matplotlib for visualization, the program provides an interactive, open-source tool for monitoring and analyzing audio signals.

---

## Features

- Real-time audio capture from system microphone
- Hanning window to reduce spectral leakage
- Fast Fourier Transform (FFT) for frequency analysis
- Dynamic, continuously updating spectrum visualization
- Lightweight and easy to use
- Cross-platform support (Windows, macOS, Linux)

---

## Requirements

- Python 3.8+
- Libraries:
  - `numpy`
  - `matplotlib`
  - `pyaudio`

Install dependencies with:

```bash
pip install numpy matplotlib pyaudio
```

*Note for macOS users:* You may need to install PortAudio first:

```bash
brew install portaudio
pip install pyaudio
```

---

## Usage

1. Connect a working microphone to your system.
2. Run the Python script:

```bash
python realtime_audio_spectrum.py
```

3. A plot window will open showing the real-time frequency spectrum. Peaks correspond to dominant frequencies.
4. Close the plot or press `Ctrl+C` to stop the program.

---

## How It Works

- Audio is captured in frames (`CHUNK` size, e.g., 1024 samples).
- A Hanning window is applied to reduce spectral leakage.
- FFT computes the magnitude spectrum for each frame.
- Matplotlib updates the plot dynamically using `FuncAnimation`.

---

## Customization

- **CHUNK size:** Change the number of samples per frame to adjust resolution.
- **RATE:** Sampling rate in Hz (default 44100).
- **Plot limits:** Adjust `ax.set_ylim()` to match expected audio levels.
- **Visualization speed:** Modify `interval` in `FuncAnimation` for faster/slower updates.

---

## Limitations

- Latency may vary depending on system performance.
- Extremely high frequencies may not display accurately due to Nyquist limit (`RATE / 2`).
- Requires a working microphone and audio input access.

---

## License

This project is released under the MIT License.