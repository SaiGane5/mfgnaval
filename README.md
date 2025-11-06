# MFG Naval Model - Full Stack Example

This repository provides a production-ready example full-stack app that wraps the Naval Propulsion Plant regression model with a FastAPI backend and a React + Vite + Tailwind frontend. The project includes Dockerfiles and a docker-compose setup for easy local development and deployment.

## Features
- FastAPI backend with prediction endpoint (mock implementation, replace with trained model)
- React (Vite) frontend styled with Tailwind CSS
- Axios-based API service
- Zustand for simple state management demo
- Docker + docker-compose for multi-container development
- Nginx reverse proxy configuration for production

## Folder structure

```
backend/
  app/
    main.py
  Dockerfile
  requirements.txt
frontend/
  src/
    pages/
    services/
  Dockerfile
  package.json
docker-compose.yml
deploy/nginx/nginx.conf
README.md
```

## Local development (recommended)

Prerequisites: Docker and Docker Compose (for containerized dev) OR Node 18+/npm and Python 3.11+.

Using Docker Compose (recommended):

1. Build and start services (hot-reload enabled via bind mounts):

```bash
docker compose up --build backend frontend-dev
```

2. Open frontend at http://localhost:5173 and backend fastapi docs at http://localhost:8000/docs

Note: The backend Docker image runs a short (`--quick`) training during image build to create a model artifact at `app/model/model.joblib` so that the `/api/predict` endpoint is available immediately after the container starts. This is convenient for dev; remove or modify this step for production deployment and use a trained artifact instead.

Using local environments (no Docker):

Backend:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

## Production (basic)

Build both images and use nginx as reverse proxy (example using docker-compose in production mode):

```bash
docker-compose -f docker-compose.yml up --build -d
```

## API

Health check:

```bash
curl http://localhost:8000/api/health
# {"status":"ok"}
```

Predict (example):

```bash
curl -X POST http://localhost:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"lp":0,"v":0,"GTT":1.2,"gtn":0,"ggn":0,"ts":0,"tp":0,"t48":500,"t1":20,"t2":30,"p48":1.0,"p1":1.0,"p2":1.0,"pexh":1.0,"tic":50,"mf":10}'
```

## Environment variables

- `FRONTEND_ORIGINS` - origin allowed by CORS for backend (default `*` in this example)
- Frontend uses `VITE_API_URL` (see `frontend/.env.example`)

## Extending for real model

- Replace the mock logic in `backend/app/main.py` with a model loader (joblib/pickle) and transform inputs like the training pipeline.
- Secure secrets and CORS for production. Use a proper production WSGI server and set `--workers` in gunicorn/uvicorn where appropriate.

### Training the model

You can train and save the model locally or trigger training via the API. The repository includes `backend/train_and_save.py` which trains a MultiOutput RandomForest pipeline and writes a joblib file to `backend/app/model/model.joblib`.

Local training:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_and_save.py --data content/data.txt --out app/model/model.joblib
```

Trigger training via API (background task):

```bash
curl -X POST http://localhost:8000/api/train
```

After training the model will be available to the `/api/predict` endpoint.

## Notes

This is a minimal but complete example intended as a starting point. It focuses on structure, containerization, and developer ergonomics. Add monitoring, rate limiting, authentication, and a proper CI workflow before production use.
