import os
from typing import Optional
from joblib import load

_here = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.getenv('MODEL_PATH', os.path.normpath(os.path.join(_here, '..', 'model', 'model.joblib')))


class ModelService:
    def __init__(self):
        self.model = None

    def load(self, path: Optional[str] = None):
        path = path or MODEL_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        self.model = load(path)
        return self.model

    def predict(self, X_df):
        if self.model is None:
            self.load()
        preds = self.model.predict(X_df)
        # expecting shape (n_samples, 2)
        return preds


model_service = ModelService()
