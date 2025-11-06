from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging
import os
from typing import Optional
import sys

from .services.model_service import model_service

logger = logging.getLogger("mfg_backend")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="MFG Naval Model API", version="0.1.0")

# Allow CORS from frontend during development - configure via env in production
origins = os.getenv("FRONTEND_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origins] if origins != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    lp: float
    v: float
    gtt: float = Field(..., alias="GTT")
    gtn: float
    ggn: float
    ts: float
    tp: float
    t48: float
    t1: float
    t2: float
    p48: float
    p1: float
    p2: float
    pexh: float
    tic: float
    mf: float


class Suggestion(BaseModel):
    severity: str
    message: str


class PredictResponse(BaseModel):
    gt_c_decay: float
    gt_t_decay: float
    suggestions: list[Suggestion] = []


@app.on_event("startup")
def load_model_on_start():
    # Try to load model at startup; ignore if missing (allow separate training step)
    try:
        model_service.load()
        logger.info("Model loaded at startup")
    except Exception as exc:
        logger.warning("No model loaded at startup: %s", exc)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get('/api/metadata')
def metadata():
    """Return feature and output metadata so frontends can render labels/units.

    The dataset describes 16 input features and two outputs (compressor and turbine
    decay coefficients). We include a short label, units and a longer description.
    """
    features = [
        {"key": "lp", "label": "Lever position", "unit": "", "description": "Lever position"},
        {"key": "v", "label": "Ship speed", "unit": "knots", "description": "Ship speed (linear function of lever position)"},
        {"key": "GTT", "label": "Gas Turbine shaft torque", "unit": "kN m", "description": "Gas Turbine (GT) shaft torque"},
        {"key": "gtn", "label": "GT rate of revolutions", "unit": "rpm", "description": "Gas Turbine rate of revolutions"},
        {"key": "ggn", "label": "Gas Generator rate of revolutions", "unit": "rpm", "description": "Gas Generator rate of revolutions"},
        {"key": "ts", "label": "Starboard Propeller Torque", "unit": "kN", "description": "Starboard Propeller Torque"},
        {"key": "tp", "label": "Port Propeller Torque", "unit": "kN", "description": "Port Propeller Torque"},
        {"key": "t48", "label": "HP Turbine exit temperature", "unit": "°C", "description": "High Pressure Turbine exit temperature (T48)"},
        {"key": "t1", "label": "GT Compressor inlet air temperature", "unit": "°C", "description": "GT Compressor inlet air temperature (T1)"},
        {"key": "t2", "label": "GT Compressor outlet air temperature", "unit": "°C", "description": "GT Compressor outlet air temperature (T2)"},
        {"key": "p48", "label": "HP Turbine exit pressure", "unit": "bar", "description": "HP Turbine exit pressure (P48)"},
        {"key": "p1", "label": "GT Compressor inlet air pressure", "unit": "bar", "description": "GT Compressor inlet air pressure (P1)"},
        {"key": "p2", "label": "GT Compressor outlet air pressure", "unit": "bar", "description": "GT Compressor outlet air pressure (P2)"},
        {"key": "pexh", "label": "GT exhaust gas pressure", "unit": "bar", "description": "Gas Turbine exhaust gas pressure (Pexh)"},
        {"key": "tic", "label": "Turbine Injection Control", "unit": "%", "description": "Turbine Injection Control (TIC)"},
        {"key": "mf", "label": "Fuel flow", "unit": "kg/s", "description": "Fuel flow (mf)"},
    ]

    outputs = [
        {"key": "gt_c_decay", "label": "GT Compressor decay state coefficient", "description": "GT Compressor decay state coefficient (kMc)"},
        {"key": "gt_t_decay", "label": "GT Turbine decay state coefficient", "description": "GT Turbine decay state coefficient (kMt)"},
    ]

    return {"features": features, "outputs": outputs, "source": "Naval GT simulator dataset (Universit\u00e0 degli Studi di Genova)"}


@app.post("/api/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    try:
        # Use the model's internal field names (not aliases) so column names match training
        # (the training pipeline expects lowercase feature names like 'gtt')
        data = payload.dict(by_alias=False)
        logger.info("Received prediction request: %s", data)

        # Convert single sample to DataFrame-like structure expected by pipeline
        import pandas as pd
        df = pd.DataFrame([data])

        preds = model_service.predict(df)
        gt_c, gt_t = float(preds[0][0]), float(preds[0][1])

        # Run lightweight analysis to generate suggestions for users
        suggestions = model_service.analyze(df, preds)

        # Normalize suggestions to Pydantic-friendly types
        return PredictResponse(gt_c_decay=gt_c, gt_t_decay=gt_t, suggestions=suggestions)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not trained. Call /api/train first.")
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post('/api/train')
def train(background_tasks: BackgroundTasks, data_path: Optional[str] = None):
    """Trigger training in background and save the model. This endpoint assumes
    the training script `train_and_save.py` is present and will write a joblib file.
    """
    import subprocess
    # train_and_save.py lives in the backend/ directory (one level up from app/)
    train_script = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'train_and_save.py'))
    out_model = os.path.join(os.path.dirname(__file__), 'model', 'model.joblib')
    data_path = data_path or os.path.join(os.path.dirname(__file__), '..', 'content', 'data.txt')

    def _run_training():
        try:
            subprocess.check_call([sys.executable, train_script, '--data', data_path, '--out', out_model])
            # reload the model after training
            model_service.load(out_model)
            logger.info('Training complete and model loaded')
        except Exception:
            logger.exception('Training failed')

    background_tasks.add_task(_run_training)
    return {"status": "training_started", "model_path": out_model}
