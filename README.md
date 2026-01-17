# Flight Price Prediction – End-to-End ML System

This project predicts flight ticket prices using structured travel information such as airline, route, stops, journey date, and departure/arrival times.  
Beyond prediction accuracy, the primary goal is to build a **reproducible, pipeline-driven, production-oriented machine learning system** that mirrors how real teams develop, tune, track, select, and deploy models.

---

## Key Capabilities

- End-to-end **DVC pipeline** from raw data to trained models
- Config-driven experimentation and hyperparameter tuning via `params.yaml`
- Baseline and tuned models for **Random Forest** and **XGBoost**
- **MLflow experiment tracking** with metrics, parameters, artifacts, and model signatures
- Automated **best-model selection** across experiments with CSV-based reporting
- **MLflow Model Registry** for versioned and controlled model promotion
- Flask-based inference application with schema-enforced inputs
- Designed for cloud-backed workflows using **AWS S3 as a DVC remote**

---

## Project Structure

This project follows the Cookiecutter Data Science structure, extended with DVC, MLflow, and a serving layer:

- `data/` – Raw, interim, and processed datasets (DVC-tracked)
- `models/` – Trained models and experiment outputs
- `src/`
  - `data/` – Data ingestion, cleaning, transformation
  - `features/` – Feature engineering logic
  - `model/` – Training, evaluation, tuning, and selection
  - `logger.py` – Structured logging
  - `exception.py` – Custom exceptions
- `dvc.yaml` – DVC pipeline definition
- `params.yaml` – Dataset splits & hyperparameter search spaces
- `flask_app/` – Flask inference application
- `requirements.txt`

---

## Reproducible ML Pipeline (DVC)

Model development is orchestrated using a **multi-stage DVC pipeline**:

- `data_gathering` – ingest raw flight price data
- `data_cleaning` – clean and standardize raw inputs
- `data_transformation` – apply transformations and formatting
- `feature_engineering` – derive model-ready features
- `data_splitting` – train/validation/test split (configurable)
- `train_model_rf` / `train_model_xgboost` – baseline model training
- `evaluate_model` – baseline model comparison
- `hyperparameter_tuning_eval` – tuned experiments
- `best_model` – aggregate experiments and select the best model

Each stage defines explicit inputs and outputs, enabling:

- Deterministic reruns
- Clear data lineage
- Minimal recomputation when data or parameters change

Run the full pipeline:

    dvc repro

---

## Experimentation & Hyperparameter Tuning

Model experimentation is **configuration-driven**, not hardcoded.

- Baseline parameters are defined in `params.yaml`
- Hyperparameter search spaces are specified for:
  - Random Forest (`n_estimators`, `max_depth`)
  - XGBoost (`n_estimators`, `max_depth`, `learning_rate`)
- All runs are logged to **MLflow Tracking**

### Best Model Selection

A dedicated pipeline stage:

- Traverses all completed experiments
- Aggregates evaluation metrics into a structured DataFrame
- Exports results as a CSV report

This CSV can be stored locally or on **AWS S3**, allowing teams to:

- Review experiment performance centrally
- Reproduce selection decisions
- Align on which model should be promoted to production

The selected model is registered in the **MLflow Model Registry**.

---

## Model Registry & Serving

- Models are versioned and promoted via MLflow Model Registry
- Input schemas are enforced using MLflow model signatures
- A Flask application loads the registry model and serves predictions
- Feature engineering is applied at inference time to prevent training–serving skew

Run the Flask app:

    python -m flask_app.app

---

## Metrics

Models are evaluated using:

- RMSE (primary metric)
- MAE
- R²

RMSE is interpreted in real terms to understand typical prediction error rather than relying on single-point accuracy.

---

## Data & Artifact Management (DVC + AWS S3)

- DVC tracks datasets, features, models, and reports
- AWS S3 is used as a DVC remote for large artifacts
- Enables:
  - Consistent retraining across machines and CI pipelines
  - Team-wide access to identical data and model versions
  - Separation of code and large data artifacts

---

## Deployment Roadmap

Planned next steps:

- Dockerization for training and serving environments
- CI pipeline for linting, testing, and container builds
- Deployment on AWS EC2
- Health checks and basic monitoring
- Stage-based model loading (Staging → Production) with rollback support

---

## Tech Stack

- Python, pandas, NumPy
- scikit-learn, XGBoost
- DVC (pipelines + S3 remote)
- MLflow (tracking, model registry, signatures)
- DagsHub
- Flask
- YAML-based configuration
- Structured logging

---

## Summary

This project demonstrates how I design **reproducible, cloud-aware ML systems** where data, models, and decisions are versioned, auditable, and shareable across teams — not just trained in notebooks.
