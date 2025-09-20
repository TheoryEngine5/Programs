import os
import random
import time
import json
from datetime import datetime
import numpy as np
import threading
from collections import defaultdict

# For audio generation (alternative to pygame/mido)
try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

# Fixed syntax
filename = __file__  # Fixed: double underscores

# === Ultimate Real-Time Fractal Orchestra Simulator ===

class FractalOrchestra:
    def __init__(self):
        self.evolution_count = 0
        self.composition_history = []
        self.fractal_cache = {}
        self.active_instruments = []
        
    def self_evolve(self):
        """Append a random evolution line to self"""
        evolution_id = random.randint(100000, 999999)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(filename, 'a') as f:
            f.write(f"\n# Evolution step {evolution_id} at {timestamp}\n")
            f.write(f"# Mutation rate increased by {random.uniform(0.01, 0.05):.3f}\n")
        
        self.evolution_count += 1
        print(f"🧬 Evolution #{evolution_id} applied!")

    # Fractal generators with enhancements
    def sierpinski(self, n, mutation_rate=0.0):
        """Enhanced Sierpinski triangle with optional mutation"""
        if n == 0: 
            return ["*"]
        
        prev = self.sierpinski(n-1, mutation_rate)
        result = []
        space = " " * len(prev[0]) if prev else " "
        
        # Top part: duplicate side by side
        for line in prev:
            if random.random() < mutation_rate:
                line = self.mutate_line(line)
            result.append(line + " " + line)
        
        # Bottom part: centered
        for line in prev:
            if random.random() < mutation_rate:
                line = self.mutate_line(line)
            result.append(space + line + space)
        
        return result

    def cantor(self, n, line="*", mutation_rate=0.0):
        """Enhanced Cantor set"""
        if n == 0: 
            return [line]
        
        first = self.cantor(n-1, line, mutation_rate)
        if random.random() < mutation_rate:
            first = [self.mutate_line(l) for l in first]
            
        second = [" " * len(line)] * len(first)
        return first + second + first

    def dragon_curve(self, n, direction=1):
        """Dragon curve fractal"""
        if n == 0:
            return ["*"]
        
        prev = self.dragon_curve(n-1, direction)
        if direction == 1:
            return prev + ["+" * len(prev[0])] + prev[::-1]
        else:
            return prev[::-1] + ["-" * len(prev[0])] + prev

    def julia_set_ascii(self, size=20, c_real=-0.7, c_imag=0.27015):
        """ASCII representation of Julia set"""
        result = []
        for y in range(size):
            line = ""
            for x in range(size*2):
                # Map to complex plane
                real = (x - size) / (size/2)
                imag = (y - size//2) / (size/2)
                z = complex(real, imag)
                c = complex(c_real, c_imag)
                
                # Julia set iteration
                iterations = 0
                while abs(z) <= 2 and iterations < 20:
                    z = z*z + c
                    iterations += 1
                
                if iterations < 20:
                    line += "*"
                else:
                    line += " "
            result.append(line)
        return result

    def mutate_line(self, line):
        """Mutate a single line of fractal"""
        return "".join(
            "*" if c == "*" and random.random() > 0.1 else
            "*" if c == " " and random.random() < 0.05 else
            " "
            for c in line
        )

    def stochastic_mutation(self, fractal, rate=0.1):
        """Apply stochastic mutations to fractal"""
        mutated = []
        for line in fractal:
            new_line = "".join(
                "*" if c == "*" and random.random() > rate else
                "*" if c == " " and random.random() < rate/2 else
                " "
                for c in line
            )
            mutated.append(new_line)
        return mutated

    def fractal_to_frequencies(self, fractal, base_freq=220):
        """Convert fractal pattern to musical frequencies"""
        frequencies = []
        for i, line in enumerate(fractal):
            star_count = line.count("*")
            if star_count > 0:
                # Map to musical scale (pentatonic)
                scale = [1, 9/8, 5/4, 3/2, 27/16]  # Just intonation ratios
                freq = base_freq * scale[star_count % len(scale)] * (2 ** (i // 12))
                duration = max(0.1, star_count * 0.05)
                amplitude = min(0.5, star_count * 0.1)
                frequencies.append((freq, duration, amplitude))
        return frequencies

    def generate_tone(self, frequency, duration, amplitude=0.5, sample_rate=44100):
        """Generate a sine wave tone"""
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        # Add harmonics for richer sound
        wave = amplitude * (np.sin(2 * np.pi * frequency * t) + 
                           0.3 * np.sin(2 * np.pi * frequency * 2 * t) +
                           0.1 * np.sin(2 * np.pi * frequency * 3 * t))
        return wave

    def play_fractal_audio(self, fractal, base_freq=220):
        """Play fractal as audio if possible"""
        if not AUDIO_AVAILABLE:
            print("🔇 Audio not available (install sounddevice: pip install sounddevice)")
            return
        
        frequencies = self.fractal_to_frequencies(fractal, base_freq)
        
        print(f"🎵 Playing {len(frequencies)} notes...")
        for freq, duration, amplitude in frequencies:
            tone = self.generate_tone(freq, duration, amplitude)
            sd.play(tone, samplerate=44100)
            sd.wait()  # Wait for the note to finish

    def live_color_visual(self, fractals, iterations=10, mutation_rate=0.1):
        """Display evolving fractals with enhanced color-coded intensity"""
        colors = {
            'low': '\033[92m',     # green
            'med': '\033[93m',     # yellow  
            'high': '\033[91m',    # red
            'ultra': '\033[95m',   # magenta
            'reset': '\033[0m'
        }
        
        for iteration in range(iterations):
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print(f"🌊 Evolution Wave {iteration + 1}/{iterations}")
            print("=" * 60)
            
            for idx, fractal in enumerate(fractals):
                print(f"\n🎼 Instrument {idx + 1}:")
                mutated = self.stochastic_mutation(fractal, mutation_rate)
                
                for line in mutated:
                    intensity = line.count("*")
                    if intensity == 0:
                        color = colors['reset']
                    elif intensity < 3:
                        color = colors['low']
                    elif intensity < 6:
                        color = colors['med']
                    elif intensity < 10:
                        color = colors['high']
                    else:
                        color = colors['ultra']
                    
                    print(f"{color}{line}{colors['reset']}")
                
                # Update fractal for next iteration
                fractals[idx] = mutated
            
            print(f"\n🎵 Harmonic Density: {sum(sum(line.count('*') for line in f) for f in fractals)}")
            time.sleep(0.4)

    def save_composition_data(self, fractals, filename="fractal_composition.json"):
        """Save composition as structured data"""
        composition = {
            'timestamp': datetime.now().isoformat(),
            'evolution_count': self.evolution_count,
            'fractals': [
                {
                    'type': f'fractal_{i}',
                    'pattern': fractal,
                    'complexity': sum(line.count('*') for line in fractal),
                    'dimensions': (len(fractal), max(len(line) for line in fractal) if fractal else 0)
                }
                for i, fractal in enumerate(fractals)
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(composition, f, indent=2)
        
        self.composition_history.append(composition)
        return filename

    def create_fractal_symphony(self):
        """Generate a complete fractal symphony"""
        print("🎼 Generating Fractal Symphony...")
        
        # Create diverse fractals
        fractals = [
            self.stochastic_mutation(self.sierpinski(3, 0.05)),
            self.stochastic_mutation(self.cantor(4)),
            self.stochastic_mutation(self.dragon_curve(3)),
            self.julia_set_ascii(15)
        ]
        
        # Add some completely random fractals
        random_fractal = []
        for i in range(8):
            line = "".join(random.choice([" ", "*"]) for _ in range(20 + i*2))
            random_fractal.append(line)
        fractals.append(random_fractal)
        
        return fractals

    def interactive_mode(self):
        """Interactive fractal orchestra mode"""
        print("🎭 Entering Interactive Mode...")
        print("Commands: 'play', 'evolve', 'visual', 'new', 'quit'")
        
        fractals = self.create_fractal_symphony()
        
        while True:
            cmd = input("\n🎵 Orchestra> ").strip().lower()
            
            if cmd == 'quit':
                break
            elif cmd == 'play':
                if AUDIO_AVAILABLE:
                    for i, fractal in enumerate(fractals[:3]):  # Play first 3
                        print(f"🎼 Playing Instrument {i+1}...")
                        self.play_fractal_audio(fractal, 220 * (1.5 ** i))
                else:
                    print("🔇 Audio playback not available")
            elif cmd == 'evolve':
                self.self_evolve()
                fractals = [self.stochastic_mutation(f, 0.2) for f in fractals]
                print("🧬 All fractals evolved!")
            elif cmd == 'visual':
                self.live_color_visual(fractals, iterations=8)
            elif cmd == 'new':
                fractals = self.create_fractal_symphony()
                print("🆕 New symphony generated!")
            else:
                print("Unknown command. Try: play, evolve, visual, new, quit")

def main():
    print("=" * 60)
    print("🎼 ULTIMATE REAL-TIME FRACTAL ORCHESTRA SIMULATOR 🎼")
    print("=" * 60)
    
    orchestra = FractalOrchestra()
    
    # Generate initial symphony
    fractals = orchestra.create_fractal_symphony()
    
    print(f"\n🎵 Generated {len(fractals)} fractal instruments")
    
    # Visual performance
    print("\n🎭 Starting Visual Performance...")
    orchestra.live_color_visual(fractals, iterations=12, mutation_rate=0.15)
    
    # Save composition
    composition_file = orchestra.save_composition_data(fractals)
    print(f"\n💾 Composition saved as {composition_file}")
    
    # Audio playback (if available)
    if AUDIO_AVAILABLE:
        print("\n🎵 Playing Fractal Symphony...")
        for i, fractal in enumerate(fractals[:3]):  # Play first 3 to avoid too much noise
            print(f"🎼 Instrument {i+1}...")
            orchestra.play_fractal_audio(fractal, 220 * (1.2 ** i))
    
    # Self-evolution
    print("\n🧬 Applying Self-Evolution...")
    orchestra.self_evolve()
    
    # Interactive mode option
    response = input("\n🎭 Enter interactive mode? (y/n): ").strip().lower()
    if response == 'y':
        orchestra.interactive_mode()
    
    print("\n✨ Fractal Orchestra Performance Complete!")
    print("🔄 Run again for new multi-fractal symphonies with live visuals and stochastic evolution!")

if __name__ == "__main__":  # Fixed syntax
    main()
# Evolution step 197794 at 2025-09-20 12:43:51
# Mutation rate increased by 0.012

# Evolution step 200716 at 2025-09-20 12:48:38
# Mutation rate increased by 0.037
