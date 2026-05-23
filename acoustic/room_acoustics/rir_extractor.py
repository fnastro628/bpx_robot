#!/usr/bin/env python3
"""
CAP 9 — Room Impulse Response Extractor

Records all 4 mic channels simultaneously while (or just after) the bark plays.
Deconvolves the known log-chirp from each recording to obtain the Room Impulse
Response (RIR) per channel, then computes a 15-dimensional feature vector that
serves as the room's acoustic fingerprint.

Feature vector layout (15 values, all float32):
  [0]     T60   — broadband reverberation time (s)
  [1]     EDT   — early decay time (s)
  [2]     C80   — clarity index (dB)
  [3-6]   T60 per octave band: 250 / 500 / 1k / 2k Hz
  [7-12]  TDOA of first echo per mic pair (s)  — 6 pairs from 4 mics
  [13]    Spectral centroid of the RIR (Hz)
  [14]    Direct-to-reverb ratio (dB)

Usage (standalone):
  python rir_extractor.py --plot       # bark, record, plot the RIR + features
  python rir_extractor.py --save out.npy

As a library:
  from acoustic.room_acoustics.bark_signal import BarkSignal
  from acoustic.room_acoustics.rir_extractor import RIRExtractor
  bs = BarkSignal()
  rx = RIRExtractor(bs)
  features = rx.measure()   # plays bark, records, returns 15-float ndarray
"""

import argparse
import time
import numpy as np
import scipy.signal as sig
import scipy.fft as fft

from acoustic.room_acoustics.bark_signal import BarkSignal

SAMPLE_RATE    = 16000
RECORD_SEC     = 1.0          # record this long after bark starts
N_MICS         = 4
MIC_PAIRS      = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]   # 6 pairs


class RIRExtractor:
    def __init__(
        self,
        bark: BarkSignal | None = None,
        sample_rate: int = SAMPLE_RATE,
        record_sec: float = RECORD_SEC,
        mic_device: str | None = None,
    ):
        self.bark        = bark or BarkSignal(sample_rate=sample_rate)
        self.sample_rate = sample_rate
        self.record_sec  = record_sec
        self.mic_device  = mic_device

        self._rirs: np.ndarray | None = None        # (N_MICS, n_samples)
        self._features: np.ndarray | None = None    # (15,)

    # ── Public API ────────────────────────────────────────────────────────────

    def measure(self) -> np.ndarray:
        """Play bark, record, extract RIRs, return 15-feature vector."""
        recording = self._record_with_bark()
        self._rirs = self._extract_rirs(recording)
        self._features = self._compute_features(self._rirs)
        return self._features

    @property
    def rirs(self) -> np.ndarray | None:
        """Last measured RIRs, shape (N_MICS, n_samples). None before measure()."""
        return self._rirs

    @property
    def features(self) -> np.ndarray | None:
        return self._features

    # ── Recording ─────────────────────────────────────────────────────────────

    def _record_with_bark(self) -> np.ndarray:
        """Play bark and record simultaneously. Returns (n_samples, N_MICS)."""
        import sounddevice as sd

        n_record = int(self.sample_rate * self.record_sec)
        bark_sig  = self.bark.generate()

        recording = np.zeros((n_record, N_MICS), dtype=np.float32)

        def play_thread():
            sd.play(bark_sig, samplerate=self.sample_rate, device=self.bark.device)
            sd.wait()

        import threading
        t = threading.Thread(target=play_thread, daemon=True)

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=N_MICS,
            dtype="float32",
            device=self.mic_device,
        ) as stream:
            t.start()
            total = 0
            while total < n_record:
                chunk, _ = stream.read(min(512, n_record - total))
                recording[total : total + len(chunk)] = chunk
                total += len(chunk)

        t.join()
        return recording   # (n_record, N_MICS)

    # ── RIR Deconvolution ─────────────────────────────────────────────────────

    def _extract_rirs(self, recording: np.ndarray) -> np.ndarray:
        """
        Deconvolve the known chirp from each mic channel via frequency-domain
        division (spectral division with Wiener regularisation).
        Returns (N_MICS, n_rir) float32.
        """
        chirp    = self.bark.generate().astype(np.float64)
        n_rec    = recording.shape[0]
        n_fft    = fft.next_fast_len(n_rec + len(chirp) - 1)

        CHIRP_F  = fft.rfft(chirp, n=n_fft)
        eps      = 1e-6 * np.max(np.abs(CHIRP_F) ** 2)   # Wiener regulariser

        rirs = np.zeros((N_MICS, n_rec), dtype=np.float32)
        for ch in range(min(N_MICS, recording.shape[1])):
            mic_f     = fft.rfft(recording[:, ch].astype(np.float64), n=n_fft)
            H         = mic_f * np.conj(CHIRP_F) / (np.abs(CHIRP_F) ** 2 + eps)
            rir_full  = fft.irfft(H, n=n_fft)[:n_rec]
            rirs[ch]  = rir_full.astype(np.float32)

        return rirs

    # ── Feature Extraction ────────────────────────────────────────────────────

    def _compute_features(self, rirs: np.ndarray) -> np.ndarray:
        """Compute 15-dimensional feature vector from RIR array."""
        rir_mono = rirs.mean(axis=0)   # average across mics for broadband measures

        t60_broad    = self._t60(rir_mono)
        edt          = self._edt(rir_mono)
        c80          = self._c80(rir_mono)
        t60_bands    = self._t60_bands(rir_mono)
        tdoas        = self._tdoa_pairs(rirs)
        spec_centroid = self._spectral_centroid(rir_mono)
        drr          = self._direct_reverb_ratio(rir_mono)

        features = np.array(
            [t60_broad, edt, c80] + list(t60_bands) + list(tdoas) +
            [spec_centroid, drr],
            dtype=np.float32,
        )
        return features   # length 15

    # ── Acoustic Parameter Computations ──────────────────────────────────────

    def _t60(self, rir: np.ndarray, target_db: float = 60.0) -> float:
        """T60 via Schroeder backward integration."""
        energy   = rir ** 2
        schroeder = np.cumsum(energy[::-1])[::-1]
        schroeder = np.maximum(schroeder, 1e-12)
        edc_db   = 10 * np.log10(schroeder / schroeder[0])

        # Find -5 dB and -65 dB crossings for T60 estimate
        try:
            i5  = np.where(edc_db <= -5.0)[0][0]
            i65 = np.where(edc_db <= -65.0)[0][0]
            t60 = (i65 - i5) / self.sample_rate * (60.0 / (65.0 - 5.0))
        except IndexError:
            # EDC doesn't reach -65 dB — extrapolate from slope
            i5  = np.where(edc_db <= -5.0)[0]
            if len(i5) == 0:
                return 0.0
            slope  = np.polyfit(
                np.arange(i5[0], len(edc_db)) / self.sample_rate,
                edc_db[i5[0]:],
                1,
            )
            t60 = -target_db / slope[0] if slope[0] < 0 else 0.0

        return float(np.clip(t60, 0.0, 5.0))

    def _edt(self, rir: np.ndarray) -> float:
        """Early Decay Time — T60 computed from 0 to -10 dB slope."""
        energy   = rir ** 2
        schroeder = np.cumsum(energy[::-1])[::-1]
        schroeder = np.maximum(schroeder, 1e-12)
        edc_db   = 10 * np.log10(schroeder / schroeder[0])

        try:
            i0  = np.where(edc_db <= 0.0)[0][0]
            i10 = np.where(edc_db <= -10.0)[0][0]
            edt = (i10 - i0) / self.sample_rate * 6.0   # EDT → T60 equivalent
        except IndexError:
            edt = self._t60(rir)

        return float(np.clip(edt, 0.0, 5.0))

    def _c80(self, rir: np.ndarray) -> float:
        """C80 clarity index: early (0-80ms) vs late energy ratio in dB."""
        n80    = int(0.080 * self.sample_rate)
        early  = np.sum(rir[:n80] ** 2)
        late   = np.sum(rir[n80:] ** 2)
        if late < 1e-12:
            return 30.0
        return float(np.clip(10 * np.log10(early / late), -20.0, 40.0))

    def _t60_bands(self, rir: np.ndarray) -> list[float]:
        """T60 in four octave bands centred at 250, 500, 1000, 2000 Hz."""
        centres = [250, 500, 1000, 2000]
        results = []
        for fc in centres:
            lo  = fc / np.sqrt(2)
            hi  = min(fc * np.sqrt(2), self.sample_rate / 2 - 1)
            sos = sig.butter(4, [lo, hi], btype="band", fs=self.sample_rate, output="sos")
            filtered = sig.sosfilt(sos, rir)
            results.append(self._t60(filtered))
        return results

    def _tdoa_pairs(self, rirs: np.ndarray) -> list[float]:
        """GCC-PHAT TDOA (s) for each of the 6 mic pairs."""
        tdoas = []
        n = rirs.shape[1]
        n_fft = fft.next_fast_len(2 * n)
        for i, j in MIC_PAIRS:
            Xi = fft.rfft(rirs[i].astype(np.float64), n=n_fft)
            Xj = fft.rfft(rirs[j].astype(np.float64), n=n_fft)
            R  = Xi * np.conj(Xj)
            denom = np.abs(R)
            denom = np.maximum(denom, 1e-12)
            gcc   = fft.irfft(R / denom, n=n_fft)
            # Search ±10 ms window around zero
            max_lag = int(0.010 * self.sample_rate)
            lags    = np.concatenate([gcc[-max_lag:], gcc[:max_lag]])
            peak    = np.argmax(lags) - max_lag
            tdoas.append(float(peak / self.sample_rate))
        return tdoas

    def _spectral_centroid(self, rir: np.ndarray) -> float:
        """Frequency-weighted centroid of the RIR magnitude spectrum (Hz)."""
        spectrum = np.abs(fft.rfft(rir.astype(np.float64)))
        freqs    = np.linspace(0, self.sample_rate / 2, len(spectrum))
        denom    = np.sum(spectrum)
        if denom < 1e-12:
            return 0.0
        centroid = float(np.dot(freqs, spectrum) / denom)
        return float(np.clip(centroid, 0.0, self.sample_rate / 2))

    def _direct_reverb_ratio(self, rir: np.ndarray) -> float:
        """Direct-to-reverb ratio (dB). Direct = first 5 ms."""
        n5ms    = int(0.005 * self.sample_rate)
        direct  = np.sum(rir[:n5ms] ** 2)
        reverb  = np.sum(rir[n5ms:] ** 2)
        if reverb < 1e-12:
            return 30.0
        return float(np.clip(10 * np.log10(direct / reverb), -20.0, 40.0))


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(description="Bark + RIR measurement")
    parser.add_argument("--plot", action="store_true", help="Plot RIR and features")
    parser.add_argument("--save", metavar="FILE", help="Save features to .npy file")
    args = parser.parse_args()

    bark = BarkSignal()
    rx   = RIRExtractor(bark)

    print("Measuring RIR — barking now...")
    features = rx.measure()

    labels = [
        "T60 (s)", "EDT (s)", "C80 (dB)",
        "T60@250", "T60@500", "T60@1k", "T60@2k",
        "TDOA 0-1", "TDOA 0-2", "TDOA 0-3", "TDOA 1-2", "TDOA 1-3", "TDOA 2-3",
        "Spec centroid (Hz)", "DRR (dB)",
    ]
    print("\nFeature vector:")
    for label, val in zip(labels, features):
        print(f"  {label:22s}  {val:.4f}")

    if args.save:
        np.save(args.save, features)
        print(f"\nSaved to {args.save}")

    if args.plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(N_MICS, 1, figsize=(12, 8), sharex=True)
        t = np.arange(rx.rirs.shape[1]) / SAMPLE_RATE * 1000  # ms
        for ch, ax in enumerate(axes):
            ax.plot(t, rx.rirs[ch], linewidth=0.5)
            ax.set_ylabel(f"Mic {ch}")
        axes[-1].set_xlabel("Time (ms)")
        fig.suptitle("Room Impulse Response — 4 channels")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    _cli()
