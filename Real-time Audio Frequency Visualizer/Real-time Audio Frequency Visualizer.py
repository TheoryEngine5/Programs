import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Audio parameters
CHUNK = 1024
RATE = 44100

# Initialize PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, 
                channels=1, 
                rate=RATE, 
                input=True, 
                frames_per_buffer=CHUNK)

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(0, CHUNK // 2)  # Only need half for real FFT
line, = ax.plot(x, np.random.rand(CHUNK // 2))

# Configure plot appearance
ax.set_xlim(0, CHUNK // 2)
ax.set_ylim(0, 5000)  # Adjust based on your audio levels
ax.set_xlabel('Frequency Bin')
ax.set_ylabel('Amplitude')
ax.set_title('Real-time Audio Frequency Spectrum')

def update(frame):
    try:
        # Read audio data
        data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), 
                           dtype=np.int16)
        
        # Apply window function to reduce spectral leakage
        windowed = data * np.hanning(len(data))
        
        # Compute FFT
        freqs = np.fft.rfft(windowed)
        
        # Update plot with magnitude spectrum
        line.set_ydata(np.abs(freqs))
        
        return line,
    except Exception as e:
        print(f"Audio error: {e}")
        return line,

# Create animation
ani = FuncAnimation(fig, update, interval=50, blit=True)

try:
    plt.show()
except KeyboardInterrupt:
    print("Stopping...")
finally:
    # Clean up
    stream.stop_stream()
    stream.close()
    p.terminate()