# Demand Forecasting Platform

Production-oriented machine learning service for demand forecasting, built with Python and FastAPI.

## Overview
This project demonstrates an end-to-end forecasting workflow including:
- feature engineering from historical sales data
- model training and persistence
- real-time inference API
- batch prediction pipeline
- Dockerized deployment
- CI pipeline for automated testing

## Architecture
- **Training pipeline**: `pipelines/train.py`
- **Inference API**: `app/main.py`
- **Prediction service**: `app/predictor.py`
- **Batch scoring**: `pipelines/batch_predict.py`

## Tech Stack
- Python
- FastAPI
- scikit-learn
- pandas
- Docker
- GitHub Actions

## Run Locally

### 1. Install dependencies
```bash
pip install -r requirements.txt

### 2. Train Model
```bash
python pipelines/train.py

### 3. Start API
```bash
uvicorn app.main:app --reload

### 4. Test Health End-Point
```bash
curl http://127.0.0.1:8000/health

### 5. Test prediction endpoint
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d @sample_request.json


### Run with Docker
docker build -t demand-forecasting-platform .
docker run -p 8000:8000 demand-forecasting-platform




