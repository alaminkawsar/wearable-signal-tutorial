"""Shared helpers for the PPG tutorial series (ppg-signal/).

Provides: real-data loader for datasets/meal_related_ppg, a synthetic pulse
simulator, and hand-rolled (pedagogical) implementations of filtering, peak
detection, HRV, pulse-wave analysis, signal-quality and respiration methods.
Only numpy / pandas / scipy / matplotlib are required.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sg

REPO = Path("/home/al-amin-kawsar/Desktop/kawsar/Innovations/Projects/CGM_IR")
DATA_DIR = REPO / "datasets" / "meal_related_ppg"
GROUPS = {"before-lunch": "before", "after-meal": "after"}
PULSE_BAND = (0.7, 4.0)          # Hz, plausible pulse band (42-240 bpm)
RESP_BAND = (0.1, 0.45)          # Hz, plausible breathing band (6-27 /min)


# --------------------------------------------------------------- real data
@dataclass
class Rec:
    key: str                      # "<group>/<file stem>"
    label: str                    # "before" | "after"
    fs: float                     # Hz, from median slot period
    t: np.ndarray                 # seconds from file start
    x: np.ndarray                 # raw ADC counts, NaN in unbridged holes
    bad: np.ndarray               # bool: dropout sample (value<=0 or status!=Normal)


def load_recordings(max_per_group: int | None = None) -> list[Rec]:
    """Load wrist green-PPG CSVs; bridge holes <= 1 s, leave longer ones NaN."""
    recs = []
    for folder, label in GROUPS.items():
        paths = sorted((DATA_DIR / folder).glob("*.csv"))
        if max_per_group:
            paths = paths[:max_per_group]
        for p in paths:
            df = pd.read_csv(p)
            slot_s = float(np.median(np.diff(df["kernel_ts_us"].to_numpy()))) / 1e6
            fs = 1.0 / slot_s
            x = df["ppg_green"].to_numpy(float).copy()
            bad = (x <= 0) | (df["status_name"].to_numpy() != "Normal")
            x[bad] = np.nan
            x = pd.Series(x).interpolate("linear",
                                         limit=int(np.ceil(1.0 * fs))).to_numpy(float)
            recs.append(Rec(f"{folder}/{p.stem}", label, fs,
                            np.arange(len(x)) / fs, x, bad))
    return recs


def finite_run(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the longest NaN-free stretch of x (and its time axis)."""
    ok = ~np.isnan(x)
    if not ok.any():
        return np.array([]), np.array([])
    d = np.flatnonzero(np.diff(ok.astype(np.int8))) + 1
    edges = np.concatenate(([0], d, [len(x)]))
    runs = [(a, b) for a, b in zip(edges[:-1], edges[1:]) if ok[a]]
    a, b = max(runs, key=lambda ab: ab[1] - ab[0])
    return x[a:b], np.arange(b - a) / fs


def fill_nan(x: np.ndarray) -> np.ndarray:
    s = pd.Series(x)
    return s.interpolate("linear").ffill().bfill().to_numpy(float)


# --------------------------------------------------------------- simulator
def _cycle(ph: np.ndarray) -> np.ndarray:
    """One pulse cycle on phase in [0,1): P1 systolic, notch dip, P2 reflected."""
    g = lambda c, w, a: a * np.exp(-0.5 * ((ph - c) / w) ** 2)
    return g(0.28, 0.085, 1.0) + g(0.52, 0.11, 0.55) - g(0.43, 0.028, 0.20) \
        + g(0.78, 0.16, 0.10)


def synth_pulse(n_sec=60.0, fs=100.0, hr=65.0, rr=0.25, jitter=0.02,
                noise=0.02, amp_mod=0.12, drift=0.3, refl_shift=0.0,
                motion=0.0, motion_hz=2.2, seed=0):
    """Return (t, x, beat_times): pulse + respiration modulation + drift,
    optional motion interference (for artifact / ANC demos)."""
    rng = np.random.default_rng(seed)
    n = int(n_sec * fs)
    t = np.arange(n) / fs
    period = 60.0 / hr
    bt, tt = [], 0.3 * period
    while tt < n_sec - 0.1:
        bt.append(tt)
        tt += period + rng.normal(0, jitter * period)
    bt = np.array(bt)
    x = np.zeros(n)
    for b in bt:
        lo = int(b * fs)
        hi = min(int((b + period) * fs), n)
        if hi - lo < 3:
            continue
        ph = np.arange(hi - lo) / fs / period
        x[lo:hi] += _cycle(ph)[:hi - lo]
    resp = np.sin(2 * np.pi * rr * t + rng.uniform(0, 6.28))
    x = x * (1 + amp_mod * resp) + drift * (0.25 * np.sin(2 * np.pi * 0.017 * t + 1)
                                            + 0.15 * resp)
    if motion:
        x += motion * (np.sin(2 * np.pi * motion_hz * t + rng.normal())
                       + 0.5 * np.sin(4 * np.pi * motion_hz * t))
    return t, x + rng.normal(0, noise, n), bt


def synth_ecg(t: np.ndarray, bt: np.ndarray, delay=0.0, amp=1.2) -> np.ndarray:
    """Spiky 'ECG' (R at each beat time minus `delay` s) for PTT demos."""
    y = np.zeros_like(t)
    for b in bt:
        y += amp * np.exp(-0.5 * ((t - (b - delay)) / 0.008) ** 2)
    return y


# --------------------------------------------------------------- filtering
def bandpass(x, fs, lo=PULSE_BAND[0], hi=PULSE_BAND[1], order=4, zero_phase=True):
    b, a = sg.butter(order, [lo / (fs / 2), min(hi / (fs / 2), 0.99)], "bandpass")
    return sg.filtfilt(b, a, x) if zero_phase else sg.lfilter(b, a, x)


def baseline_median(x, fs, win_sec=1.0):
    k = max(int(win_sec * fs) // 2 * 2 + 1, 3)
    return sg.medfilt(fill_nan(x), k)


def detrend(x, fs, win_sec=1.0):
    return fill_nan(x) - baseline_median(x, fs, win_sec)


def zscore(x):
    s = np.nanstd(x)
    return (x - np.nanmean(x)) / (s if s > 0 else 1.0)


def welch(x, fs, band=(0.05, 20.0)):
    x = fill_nan(x)
    f, p = sg.welch(x, fs, nperseg=min(len(x), max(64, int(8 * fs))))
    m = (f >= band[0]) & (f <= band[1])
    return f[m], p[m]


# --------------------------------------------------------------- peaks / HR
def find_pulse_peaks(x, fs):
    xb = bandpass(x, fs)
    thr = 0.15 * np.percentile(np.abs(xb), 99)
    pk, _ = sg.find_peaks(xb, distance=int(0.4 * fs), height=thr)
    return pk


def hr_peaks(pk, fs):
    return 60.0 / np.median(np.diff(pk) / fs) if len(pk) > 2 else np.nan


def hr_autocorr(x, fs, lo=40.0, hi=180.0, nsec=8.0):
    x = bandpass(x[: int(nsec * fs)], fs)     # remove drift so acf is periodic
    x = x - x.mean()
    r = np.correlate(x, x, "full")[len(x) - 1:]
    r /= r[0] + 1e-12
    lags = np.arange(int(fs * 60 / hi), int(fs * 60 / lo))
    return 60.0 * fs / lags[np.argmax(r[lags])]


def hr_fft(x, fs, lo=40.0, hi=180.0):
    f, p = welch(x, fs, band=(lo / 60, hi / 60))
    return 60.0 * f[np.argmax(p)]


# --------------------------------------------------------------- HRV
def nn_from_peaks(pk, fs):
    return np.diff(pk) / fs


def clean_nn(nn, ratio=0.2):
    med = np.median(nn)
    bad = np.abs(nn - med) > ratio * med
    out, idx = nn.copy(), np.arange(len(nn))
    if (~bad).any():
        out[bad] = np.interp(idx[bad], idx[~bad], nn[~bad])
    return out, bad


def hrv_time(nn):
    d = np.diff(nn)
    return {"mean_hr": 60 / np.mean(nn), "sdnn": np.std(nn, ddof=1),
            "rmssd": np.sqrt(np.mean(d ** 2)),
            "pnn50": float(np.mean(np.abs(d) > 0.05)),
            "cv": np.std(nn, ddof=1) / np.mean(nn)}


# --------------------------------------------------------------- quality
def sqi_acsf(x, fs):
    """Normalised autocorrelation score in the pulse band (0..1)."""
    x = fill_nan(x[: int(8 * fs)])
    x = (x - x.mean()) / (x.std() + 1e-9)
    r = np.correlate(x, x, "full")[len(x) - 1:] / len(x)
    lags = np.arange(int(fs * 60 / 180), int(fs * 60 / 42))
    return float(np.clip(r[lags].max(), 0, 1))


def sqi_snr(x, fs):
    """Pulse fundamental + harmonics power over total power (0..1), plus f0."""
    f, p = welch(x, fs, band=(0.5, 15.0))
    fb = (f >= 0.7) & (f <= 4)
    fp, pp = f[fb], p[fb]
    if len(fp) == 0:
        return 0.0, np.nan
    f0 = fp[np.argmax(pp)]
    band = 0.08
    sig = sum(p[(f >= k * f0 - band) & (f <= k * f0 + band)].sum() for k in (1, 2, 3))
    return float(np.clip(sig / (p[f <= 15].sum() + 1e-12), 0, 1)), f0


def _beat_rows(x, fs, pk, nsamp, pre=0.25, post=0.75):
    rows = []
    for i in pk:
        lo, hi = int(i - pre * fs), int(i + post * fs)
        if lo >= 0 and hi < len(x):
            rows.append(np.interp(np.linspace(lo, hi, nsamp),
                                  np.arange(lo, hi), x[lo:hi]))
    return np.array(rows) if rows else np.empty((0, nsamp))


def beat_matrix(x, fs, nsamp=200):
    """(n_beats, nsamp) matrix, each beat min-max normalised."""
    pk = find_pulse_peaks(x, fs)
    m = _beat_rows(fill_nan(x), fs, pk, nsamp)
    mn, mx = m.min(1, keepdims=True), m.max(1, keepdims=True)
    return (m - mn) / (mx - mn + 1e-12), pk


def qqe(x, fs):
    """Quality from beat skewness/kurtosis (Elggestti 2016 style)."""
    from scipy.stats import kurtosis, skew
    xb = bandpass(x, fs)
    pk = find_pulse_peaks(x, fs)
    m = _beat_rows(xb, fs, pk, 128)
    if len(m) == 0:
        return 0.0
    vals = [0.5 * max(0, 1 - abs(skew(b) - 1.7) / 1.7)
            + 0.5 * max(0, 1 - abs(kurtosis(b, fisher=False) - 5.1) / 5.1)
            for b in m]
    return float(np.mean(vals))


# --------------------------------------------------------------- PWA
def pwa(beat, fs_eff):
    """Simplified pulse-wave analysis on one normalised beat."""
    p1 = int(np.argmax(beat))
    seg = beat[p1:p1 + max(4, len(beat) // 2)]
    notch = p1 + int(np.argmin(seg))
    after = beat[notch:]
    p2 = notch + int(np.argmax(after)) if len(after) > 3 else notch
    ri = (beat[p2] - beat[notch]) / (beat[p1] - beat[notch] + 1e-9)
    return {"p1": p1, "p2": p2, "notch": notch, "refl_index": float(ri),
            "p1p2_ms": float((p2 - p1) / fs_eff * 1000)}


# --------------------------------------------------------------- respiration
def _peak_freq(f, p, band):
    m = (f >= band[0]) & (f <= band[1])
    return 60 * f[m][np.argmax(p[m])] if m.any() else np.nan


def rr_baseline(x, fs):
    b = baseline_median(x, fs)
    f, p = welch(bandpass(b, fs, RESP_BAND[0], RESP_BAND[1], order=2), fs, RESP_BAND)
    return _peak_freq(f, p, RESP_BAND), b


def rr_am(x, fs):
    env = detrend(np.abs(sg.hilbert(bandpass(x, fs))), fs, 2.0)
    f, p = welch(env, fs, RESP_BAND)
    return _peak_freq(f, p, RESP_BAND), env


def rr_fm(x, fs):
    pk = find_pulse_peaks(x, fs)
    nn = nn_from_peaks(pk, fs)
    if len(nn) < 8:
        return np.nan, np.array([])
    t_pk = pk[1:] / fs                       # NN[i] ends at peak i+1
    if len(t_pk) < 8:
        return np.nan, np.array([])
    nn_i = np.interp(np.arange(t_pk[0], t_pk[-1], 0.25), t_pk, nn)
    f, p = welch(nn_i - nn_i.mean(), 4.0, RESP_BAND)
    return _peak_freq(f, p, RESP_BAND), nn_i


# --------------------------------------------------------------- adaptive filter
def nlms(d, u, mu=0.3, M=64):
    """NLMS noise estimate y ~ d from reference u; cleaned = d - y."""
    d, u = np.asarray(d, float), np.asarray(u, float)
    u = u / (np.sqrt(np.mean(u ** 2)) + 1e-9)
    n = len(d)
    y, w = np.zeros(n), np.zeros(M)
    for i in range(M, n):
        xk = u[i - M:i][::-1]
        y[i] = w @ xk
        w += mu * (d[i] - y[i]) * xk / (xk @ xk + 1e-6)
    return y