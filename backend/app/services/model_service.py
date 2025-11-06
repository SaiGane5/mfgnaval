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

    def analyze(self, X_df, preds):
        """Return a small list of human-readable findings / suggestions based on inputs and predictions.

        This is rule-based, lightweight, and intended to provide actionable hints to users
        (e.g., high temperature -> suggest cooling/load reduction). Keep rules conservative.
        """
        suggestions = []
        # Ensure single-sample handling
        try:
            row = X_df.iloc[0]
        except Exception:
            row = None

        # Interpret predictions
        try:
            gt_c, gt_t = float(preds[0][0]), float(preds[0][1])
        except Exception:
            gt_c, gt_t = None, None

        # Rule: High turbine exit temp suggests overheating / reduce load
        if row is not None and 't48' in row:
            t48 = float(row['t48'])
            if t48 >= 680:
                suggestions.append({'severity': 'critical', 'message': f'High turbine exit temperature (T48={t48}). Consider reducing load and inspecting cooling systems.'})
            elif t48 >= 620:
                suggestions.append({'severity': 'warn', 'message': f'Elevated T48={t48}. Monitor temperature and consider inspection if trend continues.'})

        # Rule: Very high fuel flow
        if row is not None and 'mf' in row:
            mf = float(row['mf'])
            if mf >= 15:
                suggestions.append({'severity': 'critical', 'message': f'High fuel flow (mf={mf}). Check fuel system and combustion efficiency.'})
            elif mf >= 8:
                suggestions.append({'severity': 'warn', 'message': f'Elevated fuel flow (mf={mf}). Investigate fuel economy and possible leaks.'})

        # Rule: Low compressor outlet pressure versus inlet pressure (pressure ratio low)
        if row is not None and {'p2', 'p1'}.issubset(row.index):
            try:
                p1, p2 = float(row['p1']), float(row['p2'])
                pr = p2 / max(1e-6, p1)
                if pr < 1.1:
                    suggestions.append({'severity': 'warn', 'message': f'Low compressor pressure ratio (p2/p1={pr:.2f}). Possible compressor performance degradation.'})
            except Exception:
                pass

        # Prediction-driven suggestion: if predicted decay coefficients are low
        if gt_c is not None:
            if gt_c < 0.96:
                suggestions.append({'severity': 'warn', 'message': f'Predicted compressor decay coefficient is {gt_c:.4f} — consider maintenance scheduling.'})
            if gt_c < 0.92:
                suggestions.append({'severity': 'critical', 'message': f'Compressor decay coefficient {gt_c:.4f} is low — immediate inspection recommended.'})

        if gt_t is not None:
            if gt_t < 0.975:
                suggestions.append({'severity': 'warn', 'message': f'Predicted turbine decay coefficient is {gt_t:.4f} — monitor and plan inspection.'})
            if gt_t < 0.95:
                suggestions.append({'severity': 'critical', 'message': f'Turbine decay coefficient {gt_t:.4f} is low — consider immediate diagnostics.'})

        # If no suggestions generated, give a positive hint
        if not suggestions:
            suggestions.append({'severity': 'info', 'message': 'No immediate issues detected from basic rule checks. Continue routine monitoring.'})

        return suggestions


model_service = ModelService()
