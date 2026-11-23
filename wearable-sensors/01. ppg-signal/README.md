# The PPG Signal — A Complete Practical Tutorial Series

Everything about **photoplethysmography (PPG)** — from photons in the skin to machine-learning
features — written for the wrist green-PPG data in this repo
(`datasets/meal_related_ppg/`, 13 recordings, ~25 Hz, one person, before/after lunch).

> No PPG background assumed. Math is optional at first pass — every claim has a figure you can
> regenerate. Python + DSP/ML maturity assumed (see the companion
> [`../dimensionality-reduction-clustering/`](../dimensionality-reduction-clustering/) tutorial).

Rebuild every figure (all deterministic, seeded):

```bash
../ppg-baseline/pulseppg/.venv/bin/python make_figures_01_06.py
../ppg-baseline/pulseppg/.venv/bin/python make_figures_07_11.py
../ppg-baseline/pulseppg/.venv/bin/python make_figures_12_18.py
```

Shared code lives in [`ppg_synth.py`](ppg_synth.py): real-data loader, a pulse-wave simulator,
and hand-rolled HRV/PWA/quality/respiration functions (hand-rolled on purpose — you learn more
and it matches the docs exactly).

## Curriculum

**Module 0 · What PPG is**

| # | Doc | Key takeaway |
|---|-----|--------------|
| 01 | [What a PPG signal is: light, blood, AC/DC](01_what_ppg_is.md) | PPG = optical measurement of **arteriolar blood volume**; pulse = 1-5% AC ripple on a huge DC pedestal |
| 02 | [Anatomy of one pulse wave](02_pulse_wave_anatomy.md) | P1, dicrotic notch, P2; the shape *is* vascular physiology |
| 03 | [Reading real PPG data: your files, sampling, dropouts](03_reading_real_ppg_data.md) | 25 Hz slots, ADC counts, `status_name`, bridge ≤1 s holes — never longer |

**Module 1 · Making it usable**

| # | Doc | Key takeaway |
|---|-----|--------------|
| [04](04_frequency_domain.md) | Frequency-domain view | Pulse 0.7-4 Hz, breathing 0.1-0.45 Hz, motion in between — the band overlap is the whole problem |
| [05](05_filtering_cleaning.md) | Filtering & cleaning | Zero-phase `filtfilt`; median-baseline > high-pass for morphology; per-window z-score |
| [06](06_signal_quality.md) | Signal quality indices | Most windows are garbage; score (SQI/ACSF/QQE) and mask before anything else |

**Module 2 · Beat-level features**

| # | Doc | Key takeaway |
|---|-----|--------------|
| [07](07_heart_rate_peaks.md) | Heart rate, honestly | 3 detectors disagree on your real data; disagreement = quality problem |
| [08](08_beat_segmentation_template.md) | Beats & template | Resample beats to a common grid; ensemble average = your cleanest signal |
| [09](09_hrv_time_domain.md) | HRV time-domain | Clean NN first or artefacts masquerade as vagal tone; RMSSD needs long windows |
| [10](10_hrv_frequency_domain.md) | HRV frequency-domain | LF/HF caveats; Lomb-Scargle beats resampling |
| [11](11_respiratory_rate.md) | Breathing from PPG alone | Baseline-, amplitude- and frequency-modulation routes agree on a good minute |

**Module 3 · Vascular & beyond**

| # | Doc | Key takeaway |
|---|-----|--------------|
| [12](12_pulse_wave_analysis.md) | Pulse-wave analysis | Reflection/stiffness indices from one averaged beat; your before/after overlap |
| [13](13_pulse_transit_time.md) | PTT, PAT, PWV, BP | Physics is real; cuffless-BP claims need calibration and mostly failed validation |
| [14](14_spo2_multiwavelength.md) | SpO₂ & multi-wavelength | Ratio-of-ratios + calibration; impossible with green-only |

**Module 4 · Artifacts**

| # | Doc | Key takeaway |
|---|-----|--------------|
| [15](15_motion_artifact.md) | Motion artifact | Pseudo-pulses at motion frequency; QC-mask → adaptive filtering (NLMS) → robust features |

**Module 5 · ML era (connects to your PulsePPG embedding work)**

| # | Doc | Key takeaway |
|---|-----|--------------|
| [16](16_feature_engineering_ml.md) | Feature engineering | Full battery + the split-design (leakage) traps that fake 95% accuracy |
| [17](17_deep_embeddings.md) | Foundation-model embeddings | What your 512-d vectors are, how to probe them, always vs handcrafted baseline |
| [18](18_meals_glucose_ppg.md) | Meals, glucose, metabolism | Real post-meal PPG physiology, study design, why negative results like yours are the norm |

Suggested paths: **foundations** 01→06 → **classic features** 07→15 → **ML** 16→18;
or for the meal research directly: 03 → 06 → 09 → 16 → 17 → 18.

## Your data at a glance (what every figure uses)

| Property | Value |
|---|---|
| Sensor | wrist, **green** reflectance PPG, `ppg_green` raw ADC counts (~46,000) |
| Sampling | slot-based ~**25 Hz** (40 ms slots from `kernel_ts_us`) |
| QC channel | `status_name` (`Normal` / wear flags); zero values = optical dropout |
| Files | `before-lunch/` 3 files (4 min each), `after-meal/` 10 files (~10 min each), 2026-11-23, assumed lunch 14:00 |
| Subject | **one person** — see leakage & power warnings in 16/18 |

## Public PPG datasets appendix

| Dataset | What it adds | Where |
|---|---|---|
| BIDMC-PPG (Moody 2017) | PPG + respiration belt + SpO₂, 125 Hz — validates tut 11 | physionet `bidmc_ppg` |
| IEEE SPGC-1 | 660 Hz research-grade PPG — morphology, tut 12 | ieee-dataport |
| TROIKA 2015 | PPG+ECG+accelerometer, AFib — tut 07/15 | physionet `ppg-af-2015` |
| PPG-DaLiA | **wrist** PPG+IMU, 8 activities, 700 Hz — closest to your data, tut 15/16 | idiap.ch |
| MIMIC-III/IV waveforms | ICU PPG + everything; huge QC demo (tut 06) | physionet |
| PulseDB | harmonised ~10 M segments, ML-ready | github `proudest/PulseDB` |
| Pulse-PPG | the contrastive foundation model behind your embeddings (tut 17) | `cgm/glucose-estimation/pulseppg_embed` |

## Glossary (one line each)

**AC** pulsatile component · **DC** non-pulsatile baseline · **P1** percussion (systolic) peak ·
**P2** reflected (tidal) wave · **dicrotic notch** aortic-valve-closure dip · **IBI/NN** inter-beat
interval (NN = normal-to-normal) · **HRV** heart-rate variability · **SDNN/RMSSD/pNN50** HRV
time-domain stats · **LF/HF** HRV spectral bands (0.04-0.15 / 0.15-0.4 Hz) · **RSA** respiratory
sinus arrhythmia · **RR** respiratory rate (context: not the R-R interval!) · **PWA** pulse-wave
analysis · **RI** reflection index · **PWV** pulse-wave velocity · **PTT/PAT** pulse arrival /
transit time · **SpO₂** blood oxygen saturation · **R ratio** ratio-of-ratios for SpO₂ ·
**MA** motion artifact · **ANC** adaptive noise cancellation · **SQI/ACSF/QQE** quality indices ·
**PSD/Welch/Lomb-Scargle** spectral estimators.

## Canonical references

- Allen J. *Photoplethysmography and its application in clinical physiological measurement.*
  Physiol Meas 2007 — the classic review; read alongside tuts 01-05.
- Elggestti et al. 2016 — PPG quality metric (QQE) used in tut 06.
- Orphanidou et al. 2015; de Chazal & Reilly — motion-artifact filtering (tut 15).
- Hossain et al. 2024, *Pulse-PPG* — the foundation model producing your 512-d vectors (tut 17).
- FDA guidance & CAST analyses on cuffless blood-pressure devices (tut 13).
- Klabunde, *Cardiovascular Physiology Concepts* (free online) — best explanation of pulse-wave
  morphology behind tuts 02/12.
- Castaneda 2018 review + Schöberl/Hermans postprandial HRV papers (tut 18).