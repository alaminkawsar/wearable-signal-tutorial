"""Figures for tutorials 07-11 (peak detection/HR, beat template, HRV,
frequency HRV, respiratory rate). Run: python make_figures_07_11.py"""
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
rng = np.random.default_rng(1)
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False})

recs = ps.load_recordings()
best = max(recs, key=lambda r: (~np.isnan(r.x)).sum())
xc, _ = ps.finite_run(best.x, best.fs)
XC = xc[: 120 * int(best.fs)]
fs = best.fs

# ------------------------------------------------- f07 peak detection
seg = xc[: 20 * int(fs)]
t = np.arange(len(seg)) / fs
xb = ps.bandpass(seg, fs)
pk = ps.find_pulse_peaks(seg, fs)
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))
ax[0].plot(t, seg - np.nanmedian(seg), lw=0.7, color="0.7", label="raw")
ax[0].plot(t, xb, lw=1.0, color="C0", label="bandpassed")
ax[0].plot(t[pk], xb[pk], "rv", ms=7, label="detected peaks")
ax[0].legend(fontsize=8)
ax[0].set(xlabel="s", title=f"real 20 s: {len(pk)} peaks -> HR "
                            f"{ps.hr_peaks(pk, fs):.0f} bpm")
tt, xs, bt = ps.synth_pulse(n_sec=20, hr=65, fs=fs, noise=0.05, seed=7)
xs *= 1 + 0.6 * np.sin(2 * np.pi * 0.07 * tt)              # slow amp fades
pk2 = ps.find_pulse_peaks(xs, fs)
truth = np.array([p for p in (bt * fs).astype(int) if p < len(xs)])
missed = truth[np.abs(truth[:, None] - pk2[None, :] / 1.0).min(1) > 0.3 * fs]
extra = pk2[np.abs(pk2[:, None] - truth[None, :]).min(1) > 0.3 * fs]
ax[1].plot(tt, xs, lw=0.9, color="C0")
ax[1].plot(tt[pk2], xs[pk2], "gv", ms=6, label=f"detected ({len(pk2)})")
ax[1].plot(tt[np.clip(missed, 0, len(xs) - 1)], xs[np.clip(missed, 0, len(xs) - 1)],
           "x", color="crimson", ms=8, mew=2, label=f"missed ({len(missed)})")
ax[1].plot(tt[extra], xs[extra], "x", color="orange", ms=8, mew=2,
           label=f"extra ({len(extra)})")
ax[1].legend(fontsize=8)
ax[1].set(xlabel="s", title="synthetic, amplitude fades: when amplitude varies, "
                            "a fixed height threshold misses beats")
fig.savefig(IMG / "f07_peaks.png"); plt.close(fig)

# ------------------------------------------------- f07 HR three methods
hop = 5
edges = range(0, len(XC) - 10 * int(fs), hop * int(fs))
rows = []
for i in edges:
    w = XC[i:i + 10 * int(fs)]
    rows.append([i / fs, ps.hr_peaks(ps.find_pulse_peaks(w, fs), fs),
                 ps.hr_autocorr(w, fs), ps.hr_fft(w, fs)])
R = np.array(rows)
fig, ax = plt.subplots(2, 1, figsize=(10.5, 6), height_ratios=[2, 1])
ax[0].plot(R[:, 0], R[:, 1], "o-", ms=3, lw=0.8, label="peak / median IBI")
ax[0].plot(R[:, 0], R[:, 2], "s-", ms=3, lw=0.8, label="autocorrelation")
ax[0].plot(R[:, 0], R[:, 3], "^-", ms=3, lw=0.8, label="spectral peak")
ax[0].legend(fontsize=8)
ax[0].set(ylabel="bpm", title="same 120 s of real data, 10 s windows, 3 HR methods: "
                             "they agree only when the signal cooperates")
spread = R[:, 1:].max(1) - R[:, 1:].min(1)
ax[1].bar(R[:, 0], spread, width=hop, color=np.where(spread > 15, "crimson", "seagreen"))
ax[1].set(xlabel="s", ylabel="max-min disagreement (bpm)",
          title="per-window disagreement: red = believe NONE of them; "
                "check quality (tut 06) before trusting any HR")
fig.savefig(IMG / "f07_hr_methods.png"); plt.close(fig)
print("f07 done")

# ------------------------------------------------- f08 beat segmentation
seg = XC[: 25 * int(fs)]
t = np.arange(len(seg)) / fs
pk = ps.find_pulse_peaks(seg, fs)
fig, ax = plt.subplots(2, 1, figsize=(11.5, 6), height_ratios=[1.2, 1])
ax[0].plot(t, ps.bandpass(seg, fs), lw=0.8, color="0.3")
for p in pk:
    ax[0].axvspan((p - 0.25) / fs, (p + 0.75) / fs, color="C0", alpha=0.12)
    ax[0].plot(t[p], ps.bandpass(seg, fs)[p], "rv", ms=5)
ax[0].set(xlabel="s", title="beat windows: peak-0.25 s .. peak+0.75 s "
                            "(shaded), on 25 s of real data")
m, _ = ps.beat_matrix(seg, fs, nsamp=200)
xx = np.linspace(0, 1, 200)
avg, sd = m.mean(0), m.std(0)
ax[1].plot(xx, avg, lw=2.2, color="C0")
ax[1].fill_between(xx, avg - 2 * sd, avg + 2 * sd, color="C0", alpha=0.3)
for row in m[rng.choice(len(m), 8, replace=False)]:
    ax[1].plot(xx, row, lw=0.5, color="0.6", alpha=0.7)
ax[1].set(xlabel="normalised beat time", ylabel="relative amplitude",
          title=f"ensemble average of {len(m)} beats (bold) - noise averaged away, "
                "pulse shape survives")
fig.savefig(IMG / "f08_beats.png"); plt.close(fig)

# ------------------------------------------------- f08 beat quality matrix
m, _ = ps.beat_matrix(XC, fs, nsamp=200)
avg = m.mean(0)
corr = np.array([np.corrcoef(b, avg)[0, 1] for b in m])
order = np.argsort(-corr)
n_good = int((corr > 0.8).sum())
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6), width_ratios=[3, 1])
im = ax[0].imshow(m[order], aspect="auto", cmap="magma", interpolation="nearest")
ax[0].axhline(n_good - 0.5, color="w", lw=1.4)
ax[0].text(210, n_good + 3, f"below line: r < 0.8 to template ({len(m) - n_good} beats)",
           fontsize=7.5, color="w")
ax[0].set(ylabel=f"beats sorted by template correlation (n={len(m)})",
          xlabel="beat phase", title="beat matrix: the messy tail at the bottom is "
                                     "real (arm move, sensor slip)")
ax[1].hist(corr, bins=25, orientation="horizontal", color="C0")
ax[1].invert_yaxis(); ax[1].set_xlabel("correlation to template")
fig.colorbar(im, ax=ax[0], fraction=0.03)
fig.savefig(IMG / "f08_matrix.png"); plt.close(fig)
print("f08 done")

# ------------------------------------------------- f09 NN series + artefacts
pk = ps.find_pulse_peaks(XC, fs)
nn = ps.nn_from_peaks(pk, fs)
nnc, bad = ps.clean_nn(nn)
fig, ax = plt.subplots(2, 1, figsize=(11.5, 6), height_ratios=[1.4, 1])
ax[0].plot(nn, "o-", ms=3, lw=0.7, color="C0")
ax[0].plot(np.flatnonzero(bad), nn[bad], "rx", ms=8, mew=2,
           label=f"flagged by 20%-of-median rule ({bad.sum()})")
ax[0].plot(nnc, lw=1.2, color="seagreen", label="interpolated (cleaned)")
ax[0].legend(fontsize=8)
ax[0].set(ylabel="NN interval (s)", title="real data: premature beat = one short NN "
                                          "+ one long NN - artefacts, not HRV")
raw_s, cln_s = ps.hrv_time(nn), ps.hrv_time(nnc)
w = 0.35; x0 = np.arange(2)
for k, key in enumerate(["sdnn", "rmssd"]):
    vals = [raw_s[key] * 1000, cln_s[key] * 1000]
    b_ = ax[1].bar(x0 + (k - 0.5) * w, vals, w, label=key.upper())
    ax[1].bar_label(b_, fmt="%.0f ms", fontsize=8)
ax[1].set_xticks(x0, ["raw NN", "cleaned NN"])
ax[1].legend(fontsize=8)
ax[1].set(title="RMSSD falls a lot after cleaning, SDNN little: one ectopic beat "
                "inflates the 'vagal' index most", ylabel="ms")
fig.savefig(IMG / "f09_nn.png"); plt.close(fig)

# ------------------------------------------------- f09 window length lottery
tt, xs, bt = ps.synth_pulse(n_sec=300, hr=65, rr=0.25, jitter=0.03, seed=7)
nn_s = ps.nn_from_peaks((bt * 100).astype(int), 100)
t_pk = bt[1:]
rm, pos = [], []
for L in (15, 30, 60):
    for a0 in range(0, 300 - L, 30):
        sel = nn_s[(t_pk >= a0) & (t_pk < a0 + L)]
        if len(sel) > 6:
            rm.append(np.sqrt(np.mean(np.diff(sel) ** 2)) * 1000)
            pos.append(L)
rm, pos = np.array(rm), np.array(pos)
full_rm = np.sqrt(np.mean(np.diff(nn_s) ** 2)) * 1000
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
for L in (15, 30, 60):
    ax[0].scatter(np.full((pos == L).sum(), L) + rng.normal(0, 1.2, (pos == L).sum()),
                  rm[pos == L], alpha=0.6, s=18)
ax[0].axhline(full_rm, color="k", lw=1.5, ls="--", label=f"5-min RMSSD = {full_rm:.0f} ms")
ax[0].set_xticks([15, 30, 60], ["15 s", "30 s", "60 s"])
ax[0].set(xlabel="window length (s)", ylabel="RMSSD (ms)",
          title="short-window RMSSD is a lottery: spread shrinks\nas the window grows")
ax[0].legend(fontsize=8)
nn_e = nn_s.copy()
for i in (40, 150, 220):
    nn_e[i] *= 0.6
    nn_e[i + 1] *= 1.45
ax[1].plot(nn_s, lw=0.8, color="C0", label="clean (true HRV)")
ax[1].plot(nn_e, lw=0.8, color="C3", label="3 ectopic beats added")
ax[1].text(0.5, 0.82, f"RMSSD clean  = {np.sqrt(np.mean(np.diff(nn_s)**2))*1000:.0f} ms\n"
           f"RMSSD ectopic  = {np.sqrt(np.mean(np.diff(nn_e)**2))*1000:.0f} ms\n"
           "(3 of 320 beats -> doubled index)", transform=ax[1].transAxes,
           fontsize=8, bbox=dict(fc="w", alpha=0.8))
ax[1].legend(fontsize=8)
ax[1].set(ylabel="NN (s)", xlabel="beat", title="why you MUST clean NN before HRV")
fig.savefig(IMG / "f09_window_size.png"); plt.close(fig)
print("f09 done")

# ------------------------------------------------- f10 Lomb vs Welch
nn = ps.clean_nn(ps.nn_from_peaks(ps.find_pulse_peaks(XC, fs), fs))[0]
t_pk = np.cumsum(nn)
f_r, p_r = ps.welch(np.interp(np.arange(0, t_pk[-1], 0.25), t_pk, nn), 4.0,
                    band=(0.01, 0.6))
fr = np.linspace(0.01, 0.6, 400)
p_l = sg.lombscargle(t_pk, nn - nn.mean(), 2 * np.pi * fr, normalize=True)
fig, ax = plt.subplots(figsize=(10.5, 4.6))
ax.axvspan(0.04, 0.15, color="#ffe0b2", alpha=0.7)
ax.text(0.095, 0.92, "LF\n0.04-0.15 Hz", ha="center", fontsize=8)
ax.axvspan(0.15, 0.4, color="#bbdefb", alpha=0.7)
ax.text(0.275, 0.92, "HF (respiratory)\n0.15-0.4 Hz", ha="center", fontsize=8)
ax.plot(f_r, p_r / p_r.max(), lw=1.4, color="C0", label="Welch on resampled NN")
ax.plot(fr, p_l / (p_l.max() + 1e-12), lw=1.4, color="C3",
        label="Lomb-Scargle on uneven NN")
ax.legend(fontsize=8)
ax.set(xlabel="Hz", ylabel="normalised power",
       title="HRV spectrum on real data: resampling to a uniform grid adds fake "
             "power,\nLomb-Scargle handles uneven NN - watch them disagree")
fig.savefig(IMG / "f10_lomb.png"); plt.close(fig)

# ------------------------------------------------- f10 RSA
tt, xs, bt = ps.synth_pulse(n_sec=60, hr=65, rr=0.25, amp_mod=0.35,
                            drift=0.4, seed=8)
nn_s = ps.nn_from_peaks((bt * 100).astype(int), 100)
t_pk = bt[1:]
hr_i = np.interp(tt, t_pk, 60 / nn_s)
resp = np.sin(2 * np.pi * 0.25 * tt)
r = np.corrcoef(hr_i[100:], resp[100:])[0, 1]
fig, ax = plt.subplots(2, 1, figsize=(10.5, 5.4), sharex=True)
ax[0].plot(tt, resp, lw=1.2, color="C2")
ax[0].set(ylabel="breathing", title=f"instantaneous HR follows breathing "
                                    f"(r = {r:.2f}): respiratory sinus arrhythmia")
ax[1].plot(tt, hr_i, lw=1.2, color="C3")
ax[1].set(xlabel="s", ylabel="HR (bpm)")
fig.savefig(IMG / "f10_rsa.png"); plt.close(fig)
print("f10 done")

# ------------------------------------------------- f11 respiratory rate
tt, xs, _ = ps.synth_pulse(n_sec=120, hr=65, rr=0.27, amp_mod=0.3,
                           drift=0.6, fs=50, seed=9)
rrb, b_tr = ps.rr_baseline(xs, 50)
rra, a_tr = ps.rr_am(xs, 50)
rrf, fm_tr = ps.rr_fm(xs, 50)
fig, ax = plt.subplots(2, 3, figsize=(12.5, 5.4))
ax[0, 0].plot(tt, b_tr - b_tr.mean(), lw=0.9, color="C2")
ax[0, 0].set_title(f"1. baseline wander\nRR = {rrb:.1f} /min", fontsize=9)
ax[0, 1].plot(tt, a_tr, lw=0.9, color="C0")
ax[0, 1].set_title(f"2. amplitude modulation (Hilbert)\nRR = {rra:.1f} /min", fontsize=9)
ax[0, 2].plot(np.arange(len(fm_tr)) / 4, fm_tr - fm_tr.mean(), lw=0.9, color="C3")
ax[0, 2].set_title(f"3. frequency modulation (NN)\nRR = {rrf:.1f} /min", fontsize=9)
for k, (sig_, sr) in enumerate([(b_tr, 50.0), (a_tr, 50.0), (fm_tr, 4.0)]):
    f, p = ps.welch(np.asarray(sig_) - np.mean(sig_), sr, band=(0.05, 0.6))
    ax[1, k].plot(f, p / p.max(), lw=1.2)
    ax[1, k].axvline(0.27, color="k", ls="--", lw=1)
    ax[1, k].set(xlabel="Hz", ylabel="norm. PSD",
                 title="spectrum peak vs true 0.27 Hz (16.2 /min)")
for a_ in ax[0].flat:
    a_.set(xlabel="s", yticks=[])
fig.suptitle("three independent ways to breathe out of one PPG "
             "(synthetic, true RR = 16.2 /min)", y=1.02)
fig.tight_layout(); fig.savefig(IMG / "f11_rr.png"); plt.close(fig)
print("f11 done - ALL FIGURES 07-11 WRITTEN")