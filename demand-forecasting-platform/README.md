# Demand Forecasting Platform

Advanced production-oriented machine learning service for demand forecasting, built with Python, FastAPI, XGBoost, and MLflow.

## Overview
This project demonstrates an end-to-end forecasting workflow including:
- configurable feature engineering
- XGBoost-based model training
- MLflow experiment tracking
- persisted model artifacts and metadata
- real-time inference API
- batch prediction pipeline
- Dockerized deployment
- CI pipeline for automated testing

## Architecture
- **Training pipeline**: `pipelines/train.py`
- **Inference API**: `app/main.py`
- **Prediction service**: `app/predictor.py`
- **Batch scoring**: `pipelines/batch_predict.py`
- **Configuration**: `configs/training_config.yaml`

## Tech Stack
- Python
- FastAPI
- XGBoost
- MLflow
- pandas
- scikit-learn
- Docker
- GitHub Actions

## Run Locally

### Install dependencies
```bash
pip install -r requirements.txt