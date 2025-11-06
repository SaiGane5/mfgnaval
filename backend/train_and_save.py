"""Train a multi-output pipeline from the provided data and save it as a joblib file.

Usage:
    python train_and_save.py --data content/data.txt --out app/model/model.joblib
"""
import argparse
import os
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# Try importing the model utilities relative to this script. If running from the repo root
# or via different import paths, prefer local module fallback.
try:
    import backend.model as model_script
except Exception:
    import model as model_script


def train_and_save(data_path: str, out_path: str, quick: bool = False):
    df = model_script.load_data(data_path)
    X, Y, _ = model_script.split_xy(df)

    num_cols = X.columns.tolist()
    pre = ColumnTransformer([('scale', StandardScaler(), num_cols)], remainder='drop')
    n_estimators = 50 if quick else 600
    base = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=model_script.RANDOM_STATE)
    multi = MultiOutputRegressor(base)
    pipe = Pipeline([('pre', pre), ('est', multi)])

    print("Training model on:", data_path)
    pipe.fit(X, Y)

    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    dump(pipe, out_path)
    print("Saved model to:", out_path)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='content/data.txt')
    ap.add_argument('--out', default='app/model/model.joblib')
    ap.add_argument('--quick', action='store_true', help='Run a short/faster training for dev/build')
    args = ap.parse_args()
    train_and_save(args.data, args.out, quick=args.quick)
