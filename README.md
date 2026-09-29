# Synthetic Insurance Claims Risk Dataset

A reproducible Python data generator for insurance-analytics prototypes. It creates 8,000 synthetic claims spanning four product categories, then engineers operational and risk-oriented features for downstream BI or modeling.

## Features

- Deterministic synthetic data generation via a seeded NumPy generator.
- Claim amounts influenced by category, severity, and seasonal effects.
- Derived loss ratio, claim duration, reporting, and calendar features.
- Exports a clean CSV to `data/claims_clean.csv`.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/generate_and_clean_claims.py
```

The data is synthetic and intended for demonstrations, exploratory analysis, and dashboard development—not actuarial decisions.
