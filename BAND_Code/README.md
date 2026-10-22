# BAND: Bayesian Adaptive Neural Dynamics

Research software for hierarchical probabilistic analysis of magnetoencephalography (MEG), cross-participant Parkinson's disease classification, predictive calibration, and sensor-level oscillatory physiology.

## Components

- `band/model.py`: multi-resolution spectral-temporal encoder, hierarchical Gaussian latents, condition-dependent stochastic temporal transition, subject effects, adversarial identity suppression, covariance penalty and marginalization for unseen subjects.
- `band/train.py`, `band/run_ablations.py`: subject-grouped model fitting, participant-level predictions and ablations on common folds.
- `band/baselines.py`: spectral logistic regression and random forest baselines.
- `band/negative_controls.py`: subject-level label permutation.
- `band/physiology.py`, `band/clinical.py`, `band/analysis.py`: beta-band power, beta burst duration, sensor maps, medication pairing and motor-score association.
- `band/robustness.py`: deterministic sensor dropouts and posterior predictions.
- `band/figures.py`: plot probability calibration and selective-risk source data from actual held-out predictions.
- `band/preprocess.py`: FIF manifest to diagnosis window arrays; must validate hardware-dependent sensor and line-noise processing before use on new data.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Diagnosis input format

Provide `data/natmeg_windows.npz` with `X` (float32 `[n_windows,n_channels,n_samples]`), `y` (integer 0=HC, 1=PD), and `subject_id` (string or integer, length `n_windows`). Each subject must have one diagnosis, sensor ordering must be consistent, and no windows from one subject may appear in both train and test folds. All MEG recordings are obtained independently from NatMEG-PD; this repository does not redistribute raw files.

```bash
python -m band.train --data data/natmeg_windows.npz --output outputs/band --folds 5 --epochs 100
python -m band.baselines --data data/natmeg_windows.npz --output outputs/baselines --folds 5
python -m band.run_ablations --data data/natmeg_windows.npz --output outputs/ablations --folds 5 --epochs 100
python -m band.figures --predictions outputs/band/subject_predictions.csv --output outputs/figures
python -m band.analysis physiology --data data/natmeg_windows.npz --output outputs/physiology
```

## Medication and clinical tables

Create a CSV where each row is one participant/medication-state measurement with `subject_id,medication,beta_power` (medication values `OFF`/`ON`). Then run:

```bash
python -m band.analysis medication --csv data/medication_features.csv --output outputs/medication
```

For symptom associations create a CSV with `subject_id,mds_updrs_iii,beta_power`, and run:

```bash
python -m band.analysis association --csv data/clinical_features.csv --output outputs/association
```

