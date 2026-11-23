"""Figures for tutorials 01-06 (physics, morphology, real data, spectra,
filtering, quality). Run: python make_figures_01_06.py"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sg

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import ppg_synth as ps

IMG = HERE / "images"
rng = np.random.default_rng(0)
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False})

recs = ps.load_recordings()
best = max(recs, key=lambda r: (~np.isnan(r.x)).sum())      # longest clean run
xc, _ = ps.finite_run(best.x, best.fs)
XC = xc[: 120 * int(best.fs)]                                # 120 s reference seg

# ------------------------------------------------- f01 light path schematic
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))
a = ax[0]
for name, col, y0 in [("epidermis", "#f7d9a0", 0.0), ("dermis", "#f2a9b0", 0.22),
                      ("subcutis (fat)", "#f5e6a8", 0.52)]:
    a.add_patch(plt.Rectangle((0, y0), 1, 0.30, color=col, alpha=0.8))
    a.text(1.02, y0 + 0.15, name, fontsize=8, va="center")
for xar in (0.35, 0.65):                                     # arterioles
    a.add_patch(plt.Circle((xar, 0.38), 0.045, color="crimson"))
a.text(0.5, 0.315, "dermal arterioles (pulsatile blood)", fontsize=7.5,
       ha="center", color="darkred")
a.add_patch(plt.Rectangle((0.2, 1.0), 0.25, 0.12, color="0.25"))      # LED
a.add_patch(plt.Rectangle((0.6, 1.0), 0.25, 0.12, color="#0a66c2"))   # PD
a.text(0.325, 1.06, "green LED\n525-540 nm", ha="center", va="center",
       color="w", fontsize=7)
a.text(0.725, 1.06, "photodetector", ha="center", va="center", color="w", fontsize=7)
a.annotate("", xy=(0.45, 0.02), xytext=(0.32, 1.0),
           arrowprops=dict(arrowstyle="-|>", color="green", lw=2))
a.annotate("", xy=(0.72, 1.0), xytext=(0.5, 0.42),
           arrowprops=dict(arrowstyle="-|>", color="green", lw=2, ls="--"))
a.text(0.02, 0.72, "light scatters through tissue;\nsystole: more blood ->\nmore absorption -> less return",
       fontsize=7.5)
a.set_xlim(-0.05, 1.55); a.set_ylim(-0.02, 1.2); a.axis("off")
a.set_title("reflectance PPG (wrist): LED shines, detector counts\n"
            "the green light that survives the round trip", fontsize=9)
b = ax[1]
t = np.linspace(0, 12, 600)
bv = 0.55 + 0.45 * ps._cycle((t * 1.1) % 1)                  # blood volume
i0 = 0.42 - 0.18 * (bv - 0.55)                               # detected intensity
b.plot(t, bv, color="crimson", label="arteriolar blood volume (AC+DC)")
b.plot(t, i0 + 2.0, color="seagreen", label="detected light intensity (inverted)")
b.axhline(2.0 + np.mean(i0), color="0.5", ls="--", lw=1)
b.text(0.1, 2.02 + np.mean(i0), "DC (tissue+venous)", fontsize=7, color="0.4")
b.annotate("AC = pulse\n(~1-5% of DC)", xy=(7.0, 2.33), xytext=(8.4, 2.5),
           fontsize=8, arrowprops=dict(arrowstyle="->"))
b.legend(fontsize=7.5, loc="lower left")
b.set(xlabel="s", ylabel="a.u.", xticks=[],
      title="DC tells you perfusion; the AC wiggle is the pulse")
fig.savefig(IMG / "f01_light_path.png"); plt.close(fig)

# ------------------------------------------------- f01 AC/DC on real data
fig, ax = plt.subplots(3, 1, figsize=(11, 6))
t_all = best.t / 60
ax[0].plot(t_all, best.x, lw=0.4, color="0.3")
ax[0].set(xlabel="minutes from file start", ylabel="raw ADC counts",
          title=f"real wrist PPG ({best.key}): DC ~46,000 counts, gaps = dropouts")
seg = xc[: 12 * int(best.fs)]
t12 = np.arange(len(seg)) / best.fs
base = ps.baseline_median(seg, best.fs)
ax[1].plot(t12, seg - np.nanmedian(seg), lw=0.8, label="raw (centred)")
ax[1].plot(t12, base - np.nanmedian(seg), lw=1.6, color="darkorange",
           label="median-filter baseline (win=1 s)")
ax[1].legend(fontsize=8)
ax[1].set(xlabel="s", title="12 s zoom: the wandering baseline (DC) is much "
                           "bigger than the pulse")
ax[2].plot(t12, seg - base, lw=0.9, color="seagreen")
ax[2].set(xlabel="s", ylabel="counts",
          title="baseline removed -> the pulse train (AC) is finally visible")
fig.savefig(IMG / "f01_acdc.png"); plt.close(fig)
print("f01 done")

# ------------------------------------------------- f02 beat anatomy
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
ph = np.linspace(0, 1, 800)
y = ps._cycle(ph)
a = ax[0]
a.plot(ph, y, lw=2, color="C3")
for x0, txt, yy in [(0.10, "systolic upstroke", 1.0), (0.28, "P1 percussion peak", 1.15),
                    (0.43, "dicrotic notch\n(aortic valve closes)", 0.68),
                    (0.52, "P2 reflected wave", 0.98), (0.85, "diastolic decay", 0.5)]:
    a.annotate(txt, xy=(x0, np.interp(x0, ph, y)), xytext=(x0 - 0.04, yy),
               fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
a.axhline(0, color="0.7", lw=0.8)
a.set(xlabel="phase of cardiac cycle", ylabel="relative blood volume",
      title="one pulse wave (idealised template)")
b = ax[1]
tt, xs, _ = ps.synth_pulse(n_sec=3.2, hr=60, rr=0.0, drift=0, noise=0.01, fs=200)
b.plot(tt, xs, lw=1.2, color="C3")
pk, _ = sg.find_peaks(xs, distance=int(0.5 * 200))
for p in pk:
    b.axvline(p / 200, color="0.6", lw=0.6, ls=":")
b.annotate("peak-to-peak distance = RR interval\n-> heart rate",
           xy=(pk[1] / 200, 1.25), xytext=(1.1, 1.5), fontsize=8,
           arrowprops=dict(arrowstyle="->"))
b.set(xlabel="s", title="the same wave repeated at heart rate;\n"
                       "fiducials: onset, P1, notch, P2")
fig.savefig(IMG / "f02_beat_anatomy.png"); plt.close(fig)

# ------------------------------------------------- f02 real overlay + aging
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
m, _ = ps.beat_matrix(XC, best.fs)
avg, sd = m.mean(0), m.std(0)
xx = np.linspace(0, 1, m.shape[1])
ax[0].plot(xx, avg, lw=2, color="C0")
ax[0].fill_between(xx, avg - 2 * sd, avg + 2 * sd, alpha=0.3, color="C0")
ax[0].set(title=f"your real data: {m.shape[0]} beats aligned & averaged\n"
                "(mean +/- 2 SD) - the ensemble pulse keeps its shape",
          xlabel="phase of cycle")
g = lambda c, w, aa: aa * np.exp(-0.5 * ((xx - c) / w) ** 2)
young = g(0.28, 0.085, 1) + g(0.52, 0.11, 0.55) - g(0.43, 0.028, 0.2)
old = g(0.28, 0.085, 1) + g(0.37, 0.11, 0.92) - g(0.34, 0.028, 0.12)
ax[1].plot(xx, young, lw=2, label="compliant artery (P2 late, small)")
ax[1].plot(xx, old, lw=2, color="C3", label="stiff artery (P2 fuses with P1)")
ax[1].legend(fontsize=8)
ax[1].set(title="why morphology matters: wave reflection returns EARLIER\n"
                "when arteries stiffen - visible in one averaged beat",
          xlabel="phase of cycle")
fig.savefig(IMG / "f02_beat_overlay.png"); plt.close(fig)
print("f02 done")

# ------------------------------------------------- f03 real-file anatomy
fig, ax = plt.subplots(2, 1, figsize=(11.5, 6.2), height_ratios=[2, 1])
a = ax[0]
a.plot(best.t / 60, best.x, lw=0.4, color="0.3")
bad = best.bad
d = np.flatnonzero(np.diff(bad.astype(np.int8))) + 1
edges = np.concatenate(([0], d, [len(bad)]))
for i0, i1 in zip(edges[:-1], edges[1:]):
    if bad[i0]:
        dur = (i1 - i0) / best.fs
        a.axvspan(i0 / best.fs / 60, i1 / best.fs / 60, color="crimson",
                  alpha=0.35 if dur > 1 else 0.15)
a.set(xlabel="min", ylabel="ADC counts",
      title=f"{best.key}: red = dropout samples (value<=0 or status!=Normal); "
            "dark = holes >1 s (never bridged)")
nb = int(best.bad.sum())
a2 = ax[1]
names = [r.key for r in recs]
drop_pct = [100 * r.bad.mean() for r in recs]
colors = ["C3" if r.label == "after" else "C0" for r in recs]
a2.barh(range(len(recs)), drop_pct, color=colors, alpha=0.8)
a2.set(ylabel="file", xlabel="% dropout samples", yticks=range(len(recs)),
       yticklabels=[n.split("/")[-1][4:18] for n in names],
       title=f"dropout % per file (blue=before-lunch, red=after-meal); "
             f"total dropouts in shown file: {nb}")
a2.invert_yaxis()
fig.tight_layout(); fig.savefig(IMG / "f03_timeline.png"); plt.close(fig)

# ------------------------------------------------- f03 gap policy demo
t = np.arange(16 * 25) / 25
tt, xs, _ = ps.synth_pulse(n_sec=16, hr=70, fs=25, noise=0.03, drift=0.1, seed=4)
x = xs.copy()
x[int(3 * 25): int(3.8 * 25)] = np.nan          # 0.8 s hole  -> bridge
x[int(9 * 25): int(14 * 25)] = np.nan           # 5.0 s hole  -> keep
xg = x.copy()
xg[: int(3.8 * 25)] = ps.fill_nan(xg[: int(3.8 * 25)])   # bridge short only
import pandas as pd
xb = pd.Series(x).interpolate("linear", limit=25).to_numpy(float)
fig, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
ax[0].plot(t, x, lw=1, color="0.3")
ax[0].axvspan(3, 3.8, color="seagreen", alpha=0.25)
ax[0].axvspan(9, 14, color="crimson", alpha=0.2)
ax[0].text(3.05, 1.25, "0.8 s hole: bridge\n(pulse keeps running)", fontsize=8)
ax[0].text(9.2, 1.25, "5 s hole: DO NOT bridge\n- interpolation would invent pulses",
           fontsize=8, color="darkred")
ax[0].set(title="gap policy: bridge short holes, never long ones", ylabel="signal")
ax[1].plot(t, xb, lw=1, color="seagreen")
ax[1].plot(t, ps.bandpass(xb, 25), lw=0.8, color="C0", alpha=0.7)
ax[1].set(xlabel="s", title="after bridging: linear fill over the hole, "
                            "bandpass unaffected -> HR still valid")
fig.savefig(IMG / "f03_gaps.png"); plt.close(fig)
print("f03 done")

# ------------------------------------------------- f04 PSD of real data
f, p = ps.welch(XC, best.fs, band=(0.03, 12))
fig, ax = plt.subplots(figsize=(10.5, 4.6))
ax.semilogy(f, p, lw=1.4, color="C0")
for lo, hi, col, lab in [(0.1, 0.45, "#c8e6c9", "breathing\n6-27/min"),
                         (0.7, 4.0, "#bbdefb", "pulse band\n42-240 bpm"),
                         (4.0, 12.0, "#ffe0b2", "motion / high-freq noise")]:
    ax.axvspan(lo, hi, color=col, alpha=0.6)
    ax.text((lo + hi) / 2, p.max() * 0.55, lab, ha="center", fontsize=8)
fs_ = best.fs
ax.axvline(fs_ / 2, color="k", lw=1, ls="--")
ax.text(fs_ / 2 + 0.05, p.max() * 0.2, "Nyquist (fs=25 Hz):\n50 Hz mains can't "
        "even exist here", fontsize=7.5)
f0 = ps.sqi_snr(XC, best.fs)[1]
for k in (1, 2, 3):
    ax.axvline(k * f0, color="crimson", lw=0.7, ls=":")
ax.annotate(f"pulse fundamental ~{f0:.2f} Hz ({60*f0:.0f} bpm)\n+ harmonics",
            xy=(f0, np.interp(f0, f, p)), xytext=(2.2, p.max() * 0.08),
            fontsize=8, color="crimson", arrowprops=dict(arrowstyle="->", color="crimson"))
ax.set(xlabel="Hz", ylabel="PSD", title="where everything lives: one real 120 s "
       "recording (Welch PSD, log scale)")
fig.savefig(IMG / "f04_psd.png"); plt.close(fig)

# ------------------------------------------------- f04 spectrogram + motion
t, xq, _ = ps.synth_pulse(n_sec=60, hr=65, fs=100, seed=5)
mo = np.zeros_like(xq)
seg = slice(int(25 * 100), int(45 * 100))
mo[seg] = 0.7 * (np.sin(2 * np.pi * 2.17 * t[seg]) + 0.4 * np.sin(4 * np.pi * 2.17 * t[seg]))
xq = xq + mo
fr, tm, Sxx = sg.spectrogram(xq, 100, nperseg=256, noverlap=200)
fig, ax = plt.subplots(2, 1, figsize=(10.5, 6), height_ratios=[1, 1.6], sharex=True)
ax[0].plot(t, xq, lw=0.5, color="0.25")
ax[0].axvspan(25, 45, color="crimson", alpha=0.15)
ax[0].set(ylabel="signal", title="20 s of motion (red): amplitude explodes, "
          "but is the FREQUENCY the heart's?")
im = ax[1].pcolormesh(tm, fr, 10 * np.log10(Sxx + 1e-12), cmap="magma", vmin=-70)
ax[1].axvspan(25, 45, color="cyan", alpha=0.12)
ax[1].axhline(65 / 60, color="cyan", lw=1, ls="--")
ax[1].text(2, 65 / 60 + 0.15, "true HR 65 bpm", color="cyan", fontsize=8)
ax[1].text(35, 2.17 + 0.15, "motion 2.17 Hz ~ 130 'bpm'\n-> naive HR reads ~130 here",
           color="cyan", fontsize=8)
ax[1].set(xlabel="s", ylabel="Hz", ylim=(0, 5))
fig.colorbar(im, ax=ax, fraction=0.03, label="dB")
fig.savefig(IMG / "f04_spectrogram.png"); plt.close(fig)
print("f04 done")

# ------------------------------------------------- f05 filter chain
seg = xc[: 15 * int(best.fs)]
t15 = np.arange(len(seg)) / best.fs
raw_c = seg - np.nanmedian(seg)
base = ps.baseline_median(seg, best.fs)
ac = ps.detrend(seg, best.fs)
bp = ps.bandpass(seg - base, best.fs)
z = ps.zscore(bp)
fig, ax = plt.subplots(5, 1, figsize=(10.5, 8.6), sharex=True)
for a_, y_, lab in [(ax[0], raw_c, "1. raw, centred"),
                    (ax[1], base - np.nanmedian(seg), "2. baseline estimate\n(median filter, 1 s)"),
                    (ax[2], ac, "3. AC = raw - baseline"),
                    (ax[3], bp, "4. bandpass 0.7-4 Hz (zero-phase Butterworth)"),
                    (ax[4], z, "5. z-score per window  <-- exactly what "
                               "PulsePPG feeds the network")]:
    a_.plot(t15, y_, lw=0.8, color="seagreen" if a_ is ax[4] else "0.25")
    a_.set_ylabel(lab, fontsize=7.5)
ax[-1].set_xlabel("s")
fig.suptitle("cleaning pipeline on 15 s of your real data", y=1.005)
fig.tight_layout(); fig.savefig(IMG / "f05_filter_chain.png"); plt.close(fig)

# ------------------------------------------------- f05 phase distortion
tt, xb, _ = ps.synth_pulse(n_sec=4, hr=60, rr=0, drift=0, noise=0.02, fs=100, seed=6)
n0 = slice(int(1.0 * 100), int(3.0 * 100))
lp = ps.bandpass(xb, 100, zero_phase=False)
zp = ps.bandpass(xb, 100, zero_phase=True)
hp1 = ps.bandpass(xb - ps.baseline_median(xb, 100), 100, lo=1.0)
med = ps.detrend(xb, 100)
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
ax[0].plot(tt[n0], xb[n0], color="0.7", lw=1, label="raw")
ax[0].plot(tt[n0], lp[n0], color="C3", lw=1.5, label="one-pass lfilter")
ax[0].plot(tt[n0], zp[n0], color="C0", lw=1.5, label="filtfilt (zero-phase)")
pk0, _ = sg.find_peaks(xb[n0]); pk1, _ = sg.find_peaks(lp[n0]); pk2, _ = sg.find_peaks(zp[n0])
ax[0].annotate(f"peak shifted\n{(pk1[1]-pk0[1])/100*1000:.0f} ms!", xy=(tt[n0][pk1[1]], zp[n0][pk1[1]]),
               xytext=(1.55, 1.1), fontsize=8, color="C3",
               arrowprops=dict(arrowstyle="->", color="C3"))
ax[0].legend(fontsize=8)
ax[0].set(xlabel="s", title="causal filter shifts peaks in TIME:\nonset/P1/P2 timings become wrong\n"
                            "-> always filtfilt for morphology")
ax[1].plot(tt[n0], hp1[n0], color="C3", lw=1.5, label="high-pass 1.0 Hz")
ax[1].plot(tt[n0], med[n0], color="C0", lw=1.5, label="median baseline removal")
ax[1].plot(tt[n0], xb[n0] - xb[n0].mean(), color="0.7", lw=0.8, label="raw")
ax[1].annotate("1 Hz HP smears the\nnotch + adds ringing", xy=(1.45, hp1[n0][45]),
               xytext=(2.15, 0.6), fontsize=8, color="C3",
               arrowprops=dict(arrowstyle="->", color="C3"))
ax[1].legend(fontsize=8)
ax[1].set(xlabel="s", title="too-aggressive high-pass destroys the\n"
                            "dicrotic notch (PWA features gone)")
fig.savefig(IMG / "f05_phase.png"); plt.close(fig)
print("f05 done")

# ------------------------------------------------- f06 window ranking
fs_ = best.fs
wins = [xc[i:i + 10 * int(fs_)] for i in range(0, len(xc) - 10 * int(fs_), 10 * int(fs_))]
sc = np.array([[ps.sqi_snr(w, fs_)[0], ps.sqi_acsf(w, fs_), ps.qqe(w, fs_)]
               for w in wins if np.isfinite(w).all()])
order = np.argsort(-(sc[:, 0] + sc[:, 2]))
pick = list(order[:4]) + list(order[-4:])
fig, ax = plt.subplots(2, 4, figsize=(12.5, 4.6))
for k, wi in enumerate(pick):
    a_ = ax[k // 4, k % 4]
    w = wins[wi]
    a_.plot(np.arange(len(w)) / fs_, ps.bandpass(w, fs_), lw=0.7, color="C2" if k < 4 else "C3")
    s = sc[wi]
    a_.set_title(f"{'GOOD ' if k < 4 else 'BAD  '}snr={s[0]:.2f} acsf={s[1]:.2f} qqe={s[2]:.2f}",
                 fontsize=8)
    a_.set_xticks([])
fig.suptitle("real 10-s windows ranked by quality indices - the four worst are "
             "not even pulse-shaped", y=1.02)
fig.tight_layout(); fig.savefig(IMG / "f06_window_rank.png"); plt.close(fig)

# ------------------------------------------------- f06 SQI scatter
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))
scat = ax[0].scatter(sc[:, 0], sc[:, 1], c=sc[:, 2], cmap="viridis", s=14)
ax[0].axvline(0.5, color="0.6", ls="--", lw=0.8); ax[0].axhline(0.8, color="0.6", ls="--", lw=0.8)
ax[0].text(0.55, 0.9, "periodic motion fools\nautocorrelation: acsf HIGH,\nspectral/qqe LOW",
           fontsize=7.5)
ax[0].set(xlabel="spectral SNR index", ylabel="autocorrelation index",
          title=f"all {len(sc)} real 10-s windows (colour = morphology qqe):\n"
                "indices disagree exactly where you'd expect", )
fig.colorbar(scat, ax=ax[0], fraction=0.04, label="qqe")
th = np.arange(len(sc))
keep = (sc[:, 0] > 0.4) & (sc[:, 2] > 0.15)
ax[1].bar(th, keep.astype(int), color=np.where(keep, "seagreen", "crimson"))
ax[1].set(xlabel="window index (chronological)", ylabel="keep / drop",
          title=f"after AND-fusing indices: {keep.sum()}/{len(sc)} windows kept.\n"
                "garbage in -> confidently wrong embeddings out")
fig.savefig(IMG / "f06_sqi_scatter.png"); plt.close(fig)
print("f06 done - ALL FIGURES 01-06 WRITTEN")