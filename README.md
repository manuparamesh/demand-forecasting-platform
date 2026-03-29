# Demand Forecasting Platform

Production-grade machine learning service for demand forecasting, built with **Python, FastAPI, XGBoost, MLflow, and Docker**.

## Key Features
- Config-driven feature engineering and training pipeline
- XGBoost-based forecasting model for tabular time-series prediction
- MLflow experiment tracking and model metadata capture
- Real-time inference API with FastAPI
- Batch scoring workflow for offline forecasting
- Dockerized deployment and CI-ready repository structure

## Tech Stack
**Python, FastAPI, XGBoost, MLflow, Pandas, scikit-learn, Docker, GitHub Actions**

## Repository Structure
```text
app/         # API, schemas, predictor service, utilities
pipelines/   # training and batch prediction pipelines
configs/     # YAML-based training configuration
models/      # saved model artifacts and metadata
tests/       # API tests
data/        # sample raw and processed data

## Architecture

The platform follows a modular ML service architecture:

- **Training pipeline** for feature generation, model fitting, and artifact persistence
- **Model metadata layer** for version traceability
- **Inference service** exposing forecast predictions through REST endpoints
- **Batch scoring pipeline** for offline prediction workflows
- **Config-driven design** to separate model parameters from application code

This repository is designed as a production-oriented reference implementation for demand f
