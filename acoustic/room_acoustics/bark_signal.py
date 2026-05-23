"""
CAP 9 — Bark Signal Generator

Generates a logarithmic frequency chirp (100 Hz → 8 kHz, 300 ms) shaped to
sound like a dog bark. Plays through the robot's speaker while recording
begins simultaneously on the XVF3800 mics.

The chirp is mathematically ideal for Room Impulse Response (RIR) estimation
because it covers the full frequency range needed and can be perfectly
deconvolved from the recording.

Usage:
  from acoustic.room_acoustics.bark_signal import BarkSignal
  bs = BarkSignal()
  signal = bs.generate()    # numpy float32 array at 16 kHz
  bs.play()                 # plays through speaker
"""

import numpy as np
import sounddevice as sd


class BarkSignal:
    def __init__(
        self,
        sample_rate: int = 16000,
        duration_sec: float = 0.30,
        freq_lo: float = 100.0,
        freq_hi: float = 8000.0,
        amplitude: float = 0.7,
        device: str | None = None,
    ):
        self.sample_rate  = sample_rate
        self.duration     = duration_sec
        self.freq_lo      = freq_lo
        self.freq_hi      = freq_hi
        self.amplitude    = amplitude
        self.device       = device
        self._signal: np.ndarray | None = None

    def generate(self) -> np.ndarray:
        """Return the chirp signal as float32 ndarray (mono, [-1, 1])."""
        n      = int(self.sample_rate * self.duration)
        t      = np.linspace(0, self.duration, n, endpoint=False)
        k      = (self.freq_hi / self.freq_lo) ** (1.0 / self.duration)

        # Log (exponential) chirp: instantaneous frequency grows exponentially
        phase  = 2 * np.pi * self.freq_lo * (k ** t - 1) / np.log(k)
        chirp  = np.sin(phase)

        # Bark envelope: fast attack, moderate decay (sounds like a bark)
        attack  = np.linspace(0, 1, n // 8)
        decay   = np.exp(-3.0 * t / self.duration)
        envelope = np.concatenate([attack, decay[n // 8:]])
        envelope = envelope[:n]

        self._signal = (chirp * envelope * self.amplitude).astype(np.float32)
        return self._signal

    def play(self) -> np.ndarray:
        """Play the bark through the speaker and return the signal used."""
        if self._signal is None:
            self.generate()
        sd.play(self._signal, samplerate=self.sample_rate, device=self.device)
        sd.wait()
        return self._signal

    @property
    def num_samples(self) -> int:
        return int(self.sample_rate * self.duration)
