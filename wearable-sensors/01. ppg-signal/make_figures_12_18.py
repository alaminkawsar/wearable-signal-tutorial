"""Figures for tutorials 12-18 (PWA, PTT/PWV, SpO2, motion artifact, ML
features, embeddings, meals). Run: python make_figures_12_18.py"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal as sg

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import ppg_synth as ps

IMG = HERE / "images"
rng = np.random.default_rng(2)
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False})

recs = ps.load_recordings()
best = max(recs, key=lambda r: (~np.isnan(r.x)).sum())
xc, _ = ps.finite_run(best.x, best.fs)
XC = xc[: 120 * int(best.fs)]
fs = best.fs

# ------------------------------------------------- f12 PPG + derivatives
m, _ = ps.beat_matrix(XC, fs, nsamp=200)
beat = m.mean(0)
beat = (beat - beat.min()) / (np.ptp(beat) + 1e-9)
d1 = np.gradient(beat)
d2 = np.gradient(d1)
tt = np.linspace(0, 1, 200)
fig, ax = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
ax[0].plot(tt, beat, lw=2, color="C0")
pk2, _ = sg.find_peaks(d2, height=0.002)
for k, p in enumerate(pk2[:5]):
    ax[0].annotate("abcde"[k], xy=(tt[p], beat[p]), xytext=(tt[p] + 0.015, beat[p]),
                   fontsize=11, color="C3", weight="bold")
ax[0].set_ylabel("PPG")
ax[1].plot(tt, d1 / np.abs(d1).max(), lw=1.6, color="C1")
ax[1].set_ylabel("PPG'")
ax[2].plot(tt, d2 / np.abs(d2).max(), lw=1.6, color="C2")
for k, p in enumerate(pk2[:5]):
    ax[2].annotate("abcde"[k], xy=(tt[p], d2[p] / np.abs(d2).max()),
                   xytext=(tt[p] + 0.015, d2[p] / np.abs(d2).max() + 0.08),
                   fontsize=11, color="C3", weight="bold")
ax[2].set_ylabel("PPG''")
ax[2].set_xlabel("normalised cycle")
fig.suptitle("ensemble beat from YOUR data and its derivatives: the PPG'' "
             "a-e waves encode reflection/stiffness", y=1.0)
fig.tight_layout(); fig.savefig(IMG / "f12_derivatives.png"); plt.close(fig)

# ------------------------------------------------- f12 PWA across files
ri, pp, labs = [], [], []
for r in recs:
    xr, _ = ps.finite_run(r.x, r.fs)
    if len(xr) < 10 * r.fs:
        continue
    mb, _ = ps.beat_matrix(xr[: 60 * int(r.fs)], r.fs, nsamp=200)
    if len(mb) < 8:
        continue
    f_ = ps.pwa(mb.mean(0), 200.0)
    ri.append(f_["refl_index"])
    pp.append(100 * (f_["p2"] - f_["p1"]) / 200)
    labs.append("after-meal" if r.label == "after" else "before")
ri, pp, labs = np.array(ri), np.array(pp), np.array(labs)
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
for k, grp in enumerate(["before", "after-meal"]):
    dx = rng.normal(0, .06, int((labs == grp).sum()))
    ax[0].scatter(np.full(int((labs == grp).sum()), k) + dx, ri[labs == grp],
                  s=26, label=grp)
    ax[1].scatter(np.full(int((labs == grp).sum()), k) + dx, pp[labs == grp],
                  s=26, label=grp)
ax[0].set(ylabel="reflection index P2/P1 (from notch)")
ax[1].set(ylabel="P1->P2 delay (% of cycle)")
for a_ in ax:
    a_.set_xticks([0, 1], ["before", "after-meal"])
    a_.set_xlabel("one point = one recording (n=13, one person)")
    a_.legend(fontsize=8)
fig.suptitle("PWA indices on your before/after files: overlapping - posture/"
             "activity confounds dominate", y=1.02)
fig.tight_layout(); fig.savefig(IMG / "f12_pwa_real.png"); plt.close(fig)
print("f12 done")

# ------------------------------------------------- f13 PTT
tt, xs, bt = ps.synth_pulse(n_sec=4, hr=60, rr=0, drift=0, noise=0.02, fs=200, seed=11)
ecg = ps.synth_ecg(tt, bt, delay=0.14)
ppg = ps.bandpass(xs, 200)
r_idx = np.argmax(ecg[int(1.5 * 200):int(2.5 * 200)]) + int(1.5 * 200)
seg = ppg[r_idx:r_idx + int(0.5 * 200)]
foot = r_idx + int(np.argmin(seg)) + 20
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
a = ax[0]
a.plot(tt, ecg * 0.9 + 1.6, lw=1, color="C0")
a.plot(tt, ppg + 0.2, lw=1.2, color="C3")
a.annotate("", xy=(foot / 200, 0.75), xytext=(r_idx / 200, 0.75),
           arrowprops=dict(arrowstyle="<|-|>", color="k"))
a.text((r_idx + foot) / 2 / 200, 0.82, f"PTT = {(foot - r_idx) / 200 * 1000:.0f} ms",
       fontsize=9, ha="center")
a.plot(tt[r_idx], 1.6 + ecg[r_idx], "r^", ms=8)
a.plot(tt[foot], ppg[foot] + 0.2, "kv", ms=8)
a.set(yticks=[], xlabel="s",
      title="ECG R-peak -> PPG foot = pulse transit time\n(a watch alone can't see "
            "R: 'PAT' uses the PPG peak instead)")
b = ax[1]
ptt = np.array([170, 158, 148, 140, 133])
b.plot(ptt, [90, 100, 110, 120, 130], "o-", color="C3")
b.set(xlabel="PTT (ms)", ylabel="systolic BP (mmHg)",
      title="why PTT excites BP people: stiffer/higher-pressure arteries\n"
            "conduct the pulse FASTER -> shorter PTT, roughly log-linear")
b.text(148, 118, "needs PER-PERSON calibration\n(and still struggled in FDA-level\n"
                 "validation of cuffless BP devices)", fontsize=8,
       bbox=dict(fc="mistyrose", alpha=0.9))
fig.savefig(IMG / "f13_ptt.png"); plt.close(fig)

# ------------------------------------------------- f13 PWV physics
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
a = ax[0]
a.add_patch(mp.FancyBboxPatch((0.08, 0.42), 0.84, 0.14,
                              boxstyle="round,pad=0.02", fc="#f2a9b0", ec="darkred"))
a.annotate("", xy=(0.88, 0.49), xytext=(0.12, 0.49),
           arrowprops=dict(arrowstyle="-|>", color="k", lw=1.8))
a.text(0.5, 0.62, "PWV = L / dt", ha="center", fontsize=10)
a.text(0.10, 0.30, "heart", fontsize=9)
a.text(0.85, 0.30, "finger", fontsize=9)
ph = np.linspace(0, 1, 100)
a.plot(0.10 + 0.08 * ph, 0.06 + 0.16 * ps._cycle(ph), lw=1.4, color="C3")
a.plot(0.80 + 0.08 * ph, 0.06 + 0.16 * ps._cycle(ph), lw=1.4, color="C3", alpha=0.6)
a.text(0.10, 0.24, "t = 0", fontsize=8)
a.text(0.80, 0.24, "t = dt", fontsize=8)
a.axis("off")
a.set_title("two-site PPG (finger vs wrist, or ECG vs PPG):\nthe same pulse "
            "arrives later and steeper", fontsize=9)
b = ax[1]
E = np.linspace(0.3, 3.0, 100)
b.plot(E, np.sqrt(E), lw=2, color="C0")
b.set(xlabel="arterial wall stiffness E (arb. units)", ylabel="PWV (arb. units)",
      title="Moens-Korteweg: PWV = sqrt(E h / 2 rho R)\nstiffness x4 -> PWV only x2 "
            "(the sqrt!)\nthis is why PWA indices track BP & age")
b.text(1.9, 0.9, "h = wall thickness\nrho = blood density\nR = radius", fontsize=8)
fig.savefig(IMG / "f13_pwv.png"); plt.close(fig)
print("f13 done")

# ------------------------------------------------- f14 SpO2
lam = np.linspace(600, 1000, 200)
hb = np.exp(-((lam - 555) ** 2) / 6000) * 0.8 + 0.25 * np.exp(-((lam - 660) ** 2) / 40000) + 0.05
hbo2 = 0.25 + 0.4 * (lam > 800) * (1 - np.exp(-(lam - 780) / 90)) + 0.05
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
a = ax[0]
a.plot(lam, hb, lw=2, color="darkred", label="deoxy-Hb (Hb)")
a.plot(lam, hbo2, lw=2, color="tomato", label="oxy-Hb (HbO2)")
a.axvline(660, color="C3", ls="--", lw=1.4); a.axvline(940, color="C0", ls="--", lw=1.4)
a.text(663, 0.9, "red 660 nm\nHb absorbs MORE", fontsize=8, color="C3")
a.text(870, 0.9, "IR 940 nm\nHbO2 absorbs MORE", fontsize=8, color="C0")
a.set(xlabel="wavelength (nm)", ylabel="relative absorption",
      title="two wavelengths are enough: the chromophores cross over")
b = ax[1]
spo2 = np.linspace(70, 100, 60)
rr_ = 3.4 - 0.030 * (spo2 - 70) ** 1.05
rr_ = np.clip(rr_, 0.4, 3.4)
b.plot(rr_, spo2, lw=2, color="C0")
idx = rng.choice(len(spo2), 12, replace=False)
b.scatter(rr_[idx] + rng.normal(0, .06, 12), spo2[idx] + rng.normal(0, 1.2, 12),
          s=20, color="0.4", label="calibration points (per device!)")
b.set(xlabel="ratio-of-ratios  R = (AC/DC)_660 / (AC/DC)_940",
      ylabel="SpO2 (%)",
      title="R -> SpO2 by EMPIRICAL calibration curve\n(invented red/green pulse "
            "ratio != SpO2)")
b.legend(fontsize=8)
fig.savefig(IMG / "f14_spo2.png"); plt.close(fig)
print("f14 done")

# ------------------------------------------------- f15 motion artifact
t, x0, bt = ps.synth_pulse(n_sec=30, hr=65, fs=100, rr=0.2, seed=12)
motion = np.sin(2 * np.pi * 2.17 * t) + 0.5 * np.sin(4 * np.pi * 2.17 * t + 0.7)
xm = x0 + 1.1 * motion
acc = motion + rng.normal(0, 0.05, len(t))
fig, ax = plt.subplots(2, 2, figsize=(12, 6.4))
ax[0, 0].plot(t, x0, lw=0.7, color="seagreen")
ax[0, 0].set(title="clean PPG (65 bpm)", ylabel="PPG")
ax[0, 1].plot(t, xm, lw=0.6, color="crimson")
ax[0, 1].plot(t, 1.1 * acc, lw=0.6, color="0.5", alpha=0.8, label="accelerometer")
ax[0, 1].legend(fontsize=7)
ax[0, 1].set(title="same + walking motion (2.17 Hz): the pulse is GONE visually",
             ylabel="PPG")
for a_, y_, name in [(ax[1, 0], x0, "clean spectrum"), (ax[1, 1], xm, "corrupted spectrum")]:
    f, p = ps.welch(y_, 100, band=(0.2, 6))
    a_.semilogy(f, p, lw=1.2)
    a_.axvline(65 / 60, color="seagreen", ls="--", lw=1)
    a_.axvline(2.17, color="crimson", ls="--", lw=1)
    a_.set(xlabel="Hz", ylim=(1e-9, 1e2))
    a_.text(0.25, 30, name, fontsize=8)
ax[1, 0].text(1.7, 1e-6, "true HR peak", color="seagreen", fontsize=8)
ax[1, 1].text(1.7, 1e-6, "motion peak 1000x bigger\n-> spectral HR now reads ~130 bpm",
              color="crimson", fontsize=8)
fig.savefig(IMG / "f15_motion.png"); plt.close(fig)

# ------------------------------------------------- f15 adaptive filtering
est = ps.nlms(xm, acc, mu=0.5, M=96)
clean = xm - est
f0_, _ = ps.sqi_snr(x0, 100)
hr_b, hr_a = ps.hr_fft(xm, 100), ps.hr_fft(clean, 100)
fig, ax = plt.subplots(2, 1, figsize=(11, 5.6))
ax[0].plot(t, xm, lw=0.6, color="crimson", label="corrupted PPG")
ax[0].plot(t, est, lw=0.8, color="C0", label="NLMS motion estimate (from accel)")
ax[0].legend(fontsize=8)
ax[0].set(title=f"adaptive noise cancellation: accel as reference, NLMS learns the "
                "motion path\nHR before = {hr_b:.0f} bpm -> after = {hr_a:.0f} bpm "
                "(true 65)", ylabel="signal")
ax[1].plot(t, clean, lw=0.8, color="seagreen")
pkc = ps.find_pulse_peaks(clean, 100)
ax[1].plot(t[pkc], clean[pkc], "rv", ms=5)
ax[1].set(xlabel="s", title="residual = PPG recovered, peaks detectable again")
fig.savefig(IMG / "f15_anc.png"); plt.close(fig)
print("f15 done")

# ------------------------------------------------- f16 feature battery
def window_features(w, fs_):
    pk = ps.find_pulse_peaks(w, fs_)
    nn, _ = ps.clean_nn(ps.nn_from_peaks(pk, fs_))
    h = ps.hrv_time(nn) if len(nn) > 4 else dict(mean_hr=np.nan, sdnn=np.nan,
                                                 rmssd=np.nan)
    f, p = ps.welch(w, fs_, band=(0.05, 5))
    bp = lambda lo, hi: float(p[(f >= lo) & (f <= hi)].sum())
    mb, _ = ps.beat_matrix(w, fs_, nsamp=128)
    sk = float(np.mean([(np.mean((b - b.mean()) ** 3) / (b.std() ** 3 + 1e-9))
                        for b in mb])) if len(mb) else np.nan
    return [h["mean_hr"], h["sdnn"] * 1e3, h["rmssd"] * 1e3,
            ps.sqi_snr(w, fs_)[0], ps.sqi_acsf(w, fs_), ps.qqe(w, fs_), sk,
            bp(0.7, 1.5) / (bp(0.05, 5) + 1e-12), bp(1.5, 3) / (bp(0.05, 5) + 1e-12)]
FNAMES = ["HR", "SDNN", "RMSSD", "SQI_snr", "SQI_acsf", "QQE", "beat_skew",
          "PSD_low/total", "PSD_high/total"]
feats = []
for r in recs:
    xr, _ = ps.finite_run(r.x, r.fs)
    ws = [xr[i:i + 20 * int(r.fs)] for i in range(0, max(len(xr) - 20 * int(r.fs), 1),
                                                   20 * int(r.fs))][:4]
    rows = [window_features(w, r.fs) for w in ws if np.isfinite(w).all() and len(w) > 5 * r.fs]
    feats.append(np.nanmean(rows, 0) if rows else np.full(len(FNAMES), np.nan))
Fm = np.vstack(feats)
Z = (Fm - np.nanmean(Fm, 0)) / (np.nanstd(Fm, 0) + 1e-9)
fig, ax = plt.subplots(figsize=(11.5, 4.6))
im = ax.imshow(Z.T, cmap="coolwarm", vmin=-2, vmax=2, aspect="auto")
ax.set_xticks(range(len(recs)), [r.key.split("/")[-1][4:15] for r in recs],
              rotation=60, ha="right", fontsize=7)
ax.set_yticks(range(len(FNAMES)), FNAMES, fontsize=8)
for k, r_ in enumerate(recs):                      # group separators
    if r_.label == "after" and recs[k - 1].label == "before":
        ax.axvline(k - 0.5, color="k", lw=1.5)
ax.text(1.5, -0.9, "before-lunch (3)", fontsize=8)
ax.text(5.0, -0.9, "after-meal (10)", fontsize=8)
fig.colorbar(im, ax=ax, fraction=0.03, label="z-score across recordings")
ax.set_title("hand-crafted feature battery per file (20 s windows, quality-masked): "
             "this matrix IS the classical pipeline", y=1.02)
fig.savefig(IMG / "f16_features.png"); plt.close(fig)
print("f16 features done")

# ------------------------------------------------- f16 leakage demo
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GroupKFold, KFold
n_sub, n_win = 4, 40
subj = np.repeat(np.arange(n_sub), n_win)
lab = np.tile(np.r_[np.zeros(n_win // 2), np.ones(n_win // 2)], n_sub)
Xs = rng.normal(0, 0.7, (n_sub * n_win, 6))
Xs += subj[:, None] * np.array([3, -2.5, 2, -3])[:4][subj][:, None] * 0.9  # subject id
Xs[:, 0] += lab * 0.35                    # the TRUE (weak) meal effect
lr = LogisticRegression(max_iter=2000)
acc_rand = cross_val_score(lr, Xs, lab, cv=KFold(5, shuffle=True, random_state=0)).mean()
acc_grp = cross_val_score(lr, Xs, lab,
                          cv=GroupKFold(4), groups=subj).mean()
fig, ax = plt.subplots(figsize=(6.4, 4.4))
b_ = ax.bar(["random split\n(LEAKS same person\nin train+test)",
             "leave-one-subject-out\n(honest generalisation)"],
            [acc_rand, acc_grp], color=["crimson", "seagreen"], width=0.55)
ax.bar_label(b_, fmt="%.2f", fontsize=11)
ax.axhline(0.5, color="0.5", ls="--", lw=1)
ax.set(ylim=(0, 1), ylabel="test accuracy", title="same data, same model, "
        "different split:\n'accuracy' was memorising WHO, not WHAT")
fig.savefig(IMG / "f16_leakage.png"); plt.close(fig)
print("f16 leakage done")

# ------------------------------------------------- f17 embedding pipeline
fig, ax = plt.subplots(figsize=(12.5, 3.6))
ax.axis("off")
boxes = [(0.00, "raw PPG\n25 Hz, counts"), (0.17, "resample\n50 Hz"),
         (0.34, "window\n30 s"), (0.51, "z-score\nper window"),
         (0.68, "frozen encoder\nPulse-PPG"), (0.85, "512-d vector\nper chunk")]
for k, (x0, txt) in enumerate(boxes):
    fc = "#bbdefb" if k < 4 else "#ffe0b2"
    ax.add_patch(mp.FancyBboxPatch((x0, 0.35), 0.13, 0.32,
                                   boxstyle="round,pad=0.015",
                                   fc=fc, ec="0.3", transform=ax.transAxes))
    ax.text(x0 + 0.065, 0.51, txt, ha="center", va="center", fontsize=8,
            transform=ax.transAxes)
    if k < 5:
        ax.annotate("", xy=(x0 + 0.17, 0.51), xytext=(x0 + 0.135, 0.51),
                    xycoords=ax.transAxes, arrowprops=dict(arrowstyle="-|>", lw=1.4))
tt, xs, _ = ps.synth_pulse(n_sec=2, hr=70, fs=25, drift=0.2, seed=13)
ax.plot(np.linspace(0.005, 0.125, len(xs)) * 1, 0.18 + 0.1 * xs / xs.max(),
        lw=0.7, color="0.3", transform=ax.transAxes)
ax.text(0.42, 0.06, "training: self-supervised contrastive on 1.5M+ minutes of "
        "PPG (Pulse-PPG) - your weights are frozen; the 512-d vector is a "
        "LEARNED feature battery: HR, HRV, morphology, quality... all entangled",
        fontsize=8, transform=ax.transAxes, ha="center")
ax.set_title("the embedding pipeline behind premeal_postmeal_ppg_embedding_vis_"
             "meal_related.ipynb", y=0.98)
fig.savefig(IMG / "f17_embeddings.png"); plt.close(fig)
print("f17 schematic done")

# ------------------------------------------------- f17 probing on YOUR data
from sklearn.neighbors import KNeighborsClassifier
wins, keys = [], []
for r in recs:
    xr, _ = ps.finite_run(r.x, r.fs)
    for i in range(0, max(len(xr) - 20 * int(r.fs), 1), 20 * int(r.fs))[:4]:
        w = xr[i:i + 20 * int(r.fs)]
        if np.isfinite(w).all() and ps.sqi_acsf(w, r.fs) > 0.6:
            wins.append(w); keys.append(r.key)
keys = np.array(keys)
y = np.array([1 if "after" in k else 0 for k in keys])
psd_F, hrv_F = [], []
for w in wins:
    f, p = ps.welch(w, fs, band=(0.1, 4))
    psd_F.append(np.log10(p + 1e-12))
    pk = ps.find_pulse_peaks(w, fs)
    nn, _ = ps.clean_nn(ps.nn_from_peaks(pk, fs))
    h = ps.hrv_time(nn) if len(nn) > 4 else {"mean_hr": np.nan, "sdnn": np.nan,
                                         "rmssd": np.nan}
    hrv_F.append([h["mean_hr"], h["sdnn"], h["rmssd"], ps.sqi_acsf(w, fs),
                  ps.qqe(w, fs)])
def loo_file_acc(F):
    F = np.asarray(F, float)
    ok = np.isfinite(F).all(1)
    F, y2, k2 = F[ok], y[ok], keys[ok]
    hit = 0
    for i in range(len(F)):
        tr = k2 != k2[i]
        if tr.sum() < 2 or (y2[tr] == y2[i]).mean() in (0.0, 1.0) and len(set(y2[tr])) < 1:
            continue
        knn = KNeighborsClassifier(3, metric="cosine").fit(F[tr], y2[tr])
        hit += knn.predict(F[[i]])[0] == y2[i]
    return hit / max(len(F), 1)
def rand_acc(F):
    F = np.asarray(F, float)
    ok = np.isfinite(F).all(1)
    F, y2 = F[ok], y[ok]
    knn = KNeighborsClassifier(3, metric="cosine").fit(F, y2)   # leaks on purpose
    return knn.score(F, y2)
fig, ax = plt.subplots(figsize=(9.5, 4.4))
labels = ["HRV/quality\n(5 features)", "log-PSD\n(64 bins)"]
g = [loo_file_acc(hrv_F), loo_file_acc(psd_F)]
r_ = [rand_acc(hrv_F), rand_acc(psd_F)]
x0 = np.arange(2)
b1 = ax.bar(x0 - 0.18, r_, 0.34, label="random window split (LEAKY)", color="crimson", alpha=0.75)
b2 = ax.bar(x0 + 0.18, g, 0.34, label="leave-one-FILE-out (honest)", color="seagreen")
ax.bar_label(b1, fmt="%.2f", fontsize=8); ax.bar_label(b2, fmt="%.2f", fontsize=8)
ax.axhline(0.5, color="0.5", ls="--", lw=1)
ax.set_xticks(x0, labels); ax.set(ylim=(0, 1.05), ylabel="before/after accuracy",
    title=f"probing YOUR {len(wins)} windows: even honest leave-one-FILE-out scores high -\n"
          "but before/after files differ in time-of-day & context: the classifier may be\n"
          "reading arousal/activity/light, NOT the meal (files are confounded with time)")
ax.legend(fontsize=8)
fig.savefig(IMG / "f17_probing.png"); plt.close(fig)
print("f17 probing done")

# ------------------------------------------------- f18 meal timeline (textbook)
h = np.linspace(0, 180, 300)
glu = 5.4 + 3.2 * np.exp(-((h - 55) ** 2) / 900) - 0.4 * np.exp(-((h - 140) ** 2) / 300)
hr = 64 + 11 * (1 - np.exp(-h / 25)) * np.exp(-h / 160)
rmssd = 40 - 14 * (1 - np.exp(-h / 30)) * np.exp(-h / 200)
amp = 1 + 0.22 * (1 - np.exp(-h / 22)) * np.exp(-h / 190)
fig, ax = plt.subplots(2, 2, figsize=(11, 6.4))
ax[0, 0].plot(h, glu, color="C0", lw=2)
ax[0, 0].set(title="CGM glucose (reference): peak ~45-75 min post-meal",
             ylabel="mmol/L", xlabel="min after meal")
ax[0, 1].plot(h, hr, color="C3", lw=2)
ax[0, 1].set(title="HR rises 5-15% (splanchnic vasodilation,\n"
                   "sympathetic activation, thermogenesis)", ylabel="bpm",
             xlabel="min after meal")
ax[1, 0].plot(h, rmssd, color="C2", lw=2)
ax[1, 0].set(title="RMSSD/HF-HRV falls (vagal withdrawal)", ylabel="ms",
             xlabel="min after meal")
ax[1, 1].plot(h, amp, color="C1", lw=2)
ax[1, 1].set(title="pulse amplitude & waveform change\n(peripheral vasodilation)",
             ylabel="rel.", xlabel="min after meal")
for a_ in ax.flat:
    a_.axvline(0, color="k", lw=1)
fig.suptitle("textbook post-meal PPG physiology (typical literature magnitudes) - "
             "signal exists, but person/act effects >> meal effect", y=1.02)
fig.tight_layout(); fig.savefig(IMG / "f18_meal_timeline.png"); plt.close(fig)

# ------------------------------------------------- f18 your HR timecourse
MEAL = pd.Timestamp("2026-11-23 14:00")
fig, ax = plt.subplots(figsize=(11.5, 4.8))
for r in recs:
    t0 = pd.to_datetime(pd.read_csv(ps.DATA_DIR / Path(r.key).parent /
                                    (Path(r.key).name + ".csv"),
                                    nrows=1)["timestamp_ms"][0], unit="ms")
    xr, _ = ps.finite_run(r.x, r.fs)
    hop = 60 * int(r.fs)
    hrs, ts = [], []
    for i in range(0, len(xr) - hop, hop):
        p_ = ps.find_pulse_peaks(xr[i:i + hop], r.fs)
        v = ps.hr_peaks(p_, r.fs)
        if np.isfinite(v) and 35 < v < 200:
            hrs.append(v); ts.append(t0 + pd.Timedelta(seconds=i / r.fs))
    if hrs:
        ax.plot(ts, hrs, "o-", ms=3, lw=0.8,
                color="C3" if r.label == "after" else "C0", alpha=0.8)
        ax.plot([], [], "o-", ms=3, color="C3" if r.label == "after" else "C0",
                alpha=0.8, label=None)
ax.axvline(MEAL, color="k", lw=1.5)
ax.text(MEAL, 175, " assumed lunch 14:00", fontsize=8)
ax.set(ylabel="HR (bpm, 60 s windows)",
       title="your 13 recordings on one clock: HR per minute (blue=before-lunch, "
             "red=after-meal)\nfiles are minutes-to-hours apart, uncontrolled "
             "posture/activity -> the design, not only the signal, hides the meal effect")
ax.legend(["before-lunch", "after-meal"], fontsize=8, loc="upper right")
import matplotlib.dates as mdates
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
fig.tight_layout(); fig.savefig(IMG / "f18_hr_timecourse.png"); plt.close(fig)
print("f18 done - ALL FIGURES 12-18 WRITTEN")