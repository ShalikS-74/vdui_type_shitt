# Historical Polity Contraction Baseline

An interpretable, leakage-aware baseline for predicting a 30% or greater
territorial contraction over the next 100 years in the Equinox/Seshat data.
It is a territorial-contraction model, not a validated collapse detector.

## Setup

```bash
git submodule update --init --recursive
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The source workbook is supplied by the `data/raw` Equinox submodule. Its
license and citation information remain in that repository.

## Reproduce the pipeline

```bash
.venv/bin/python src/01_audit.py
.venv/bin/python src/02_build.py
.venv/bin/python src/03_train.py
.venv/bin/python src/04_splits.py
.venv/bin/python src/05_audit.py
.venv/bin/python src/09_historical_audit.py
.venv/bin/python src/08_data_checks.py
.venv/bin/python src/10_model_experiments.py
.venv/bin/python src/07_dashboard_exports.py
```

Run the interface with:

```bash
.venv/bin/streamlit run app.py
```

The dashboard presents historical-indicator trajectories, false-positive
reviews, coefficient contributions, and non-causal what-if probabilities.
Generated dashboard exchange files are documented in `dashboard/README.md`.

## Caution

The primary table contains 98 labelled polity-century observations and only
six 30%-loss cases. All metrics are descriptive and unstable; see
`NEXT_STEPS.md` for limitations and proposed follow-up work.
