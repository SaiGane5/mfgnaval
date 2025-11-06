import argparse, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor # Import MultiOutputRegressor

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Naval Propulsion Plant – Regression Pipeline (Leakage-free, >0.90 R²)
...
(see previous cell for full docstring – trimmed here to keep cell short)
"""

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

DEFAULT_FEATURE_NAMES = [
    'lp','v','gtt','gtn','ggn','ts','tp','t48','t1','t2','p48','p1','p2','pexh','tic','mf',
    'gt_c_decay','gt_t_decay'
]
RANDOM_STATE = 42
RF_BASE = dict(n_estimators=600, max_depth=None, min_samples_split=2, min_samples_leaf=1, n_jobs=-1, random_state=RANDOM_STATE)
GBR_BASE = dict(n_estimators=600, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE)
XGB_BASE = dict(n_estimators=800, learning_rate=0.07, max_depth=5, subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_STATE, objective="reg:squarederror", tree_method="hist")
RF_GRID = {"rf__n_estimators":[400,800], "rf__max_depth":[None,12,18], "rf__min_samples_split":[2,5], "rf__min_samples_leaf":[1,2]}
GBR_GRID = {"gbr__n_estimators":[400,800], "gbr__learning_rate":[0.03,0.07], "gbr__max_depth":[3,4]}
XGB_GRID = {"xgb__n_estimators":[600,1000], "xgb__learning_rate":[0.05,0.1], "xgb__max_depth":[4,6]}

def load_data(path, feature_names=DEFAULT_FEATURE_NAMES):
    if not os.path.exists(path): raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path, delim_whitespace=True, header=None)
    if len(df.columns) != len(feature_names):
        raise ValueError(f"Expected {len(feature_names)} columns, found {len(df.columns)}. Check file format or feature names.")
    df.columns = feature_names
    return df

def split_xy(df):
    const_cols = [c for c in df.columns if df[c].nunique() <= 1]
    const_cols = [c for c in const_cols if c not in ("gt_c_decay","gt_t_decay")]
    df = df.drop(columns=const_cols) if const_cols else df
    X = df.drop(columns=["gt_c_decay","gt_t_decay"])
    y = df[["gt_c_decay","gt_t_decay"]]
    return X, y, const_cols

def metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, mae, rmse

def print_block(title):
    print("\n" + "="*len(title)); print(title); print("="*len(title))

def run_single_target(X_train, X_test, y_train, y_test, model_name="RF", full_tune=False, n_jobs=-1):
    num_cols = X_train.columns.tolist()
    pre = ColumnTransformer([('scale', StandardScaler(), num_cols)], remainder='drop')
    if model_name == "RF":
        est, step, grid = RandomForestRegressor(**{**RF_BASE,"n_jobs":n_jobs}), "rf", RF_GRID
    elif model_name == "GBR":
        est, step, grid = GradientBoostingRegressor(**GBR_BASE), "gbr", GBR_GRID
    elif model_name == "XGB":
        if not HAS_XGB: return None
        est, step, grid = XGBRegressor(**XGB_BASE), "xgb", XGB_GRID
    else:
        raise ValueError("Unknown model_name")
    pipe = Pipeline([('pre', pre), (step, est)])
    if full_tune:
        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        gscv = GridSearchCV(pipe, grid, scoring='r2', cv=cv, n_jobs=n_jobs, verbose=0)
        gscv.fit(X_train, y_train)
        best, best_params = gscv.best_estimator_, gscv.best_params_
    else:
        best = pipe.fit(X_train, y_train); best_params = None
    y_pred = best.predict(X_test)
    r2, mae, rmse = metrics(y_test, y_pred)
    return {"model":model_name, "best_params":best_params, "r2":r2, "mae":mae, "rmse":rmse, "estimator":best}

def run_multi_output(X_train, X_test, Y_train, Y_test, model_name="RF", full_tune=False, n_jobs=-1):
    num_cols = X_train.columns.tolist()
    pre = ColumnTransformer([('scale', StandardScaler(), num_cols)], remainder='drop')
    
    if model_name == "RF":
        est = RandomForestRegressor(**{**RF_BASE,"n_jobs":n_jobs})
        step = "rf"
        # Adjust grid parameters for MultiOutputRegressor
        grid = {f"rf__estimator__{k.split('__')[1]}": v for k, v in RF_GRID.items()}
    elif model_name == "GBR":
        est = GradientBoostingRegressor(**GBR_BASE)
        step = "gbr"
        # Adjust grid parameters for MultiOutputRegressor
        grid = {f"gbr__estimator__{k.split('__')[1]}": v for k, v in GBR_GRID.items()}
    elif model_name == "XGB":
        if not HAS_XGB: return None
        est = XGBRegressor(**XGB_BASE)
        step = "xgb"
        # Adjust grid parameters for MultiOutputRegressor
        grid = {f"xgb__estimator__{k.split('__')[1]}": v for k, v in XGB_GRID.items()}
    else:
        raise ValueError("Unknown model_name")

    # Wrap the estimator in MultiOutputRegressor for multi-target handling
    multi_output_est = MultiOutputRegressor(est)
    pipe = Pipeline([('pre', pre), (step, multi_output_est)])

    if full_tune:
        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        gscv = GridSearchCV(pipe, grid, scoring='r2', cv=cv, n_jobs=n_jobs, verbose=0)
        gscv.fit(X_train, Y_train)
        best, best_params = gscv.best_estimator_, gscv.best_params_
    else:
        best = pipe.fit(X_train, Y_train); best_params = None
        
    Y_pred = best.predict(X_test)
    r2_c, mae_c, rmse_c = metrics(Y_test.iloc[:,0].values, Y_pred[:,0])
    r2_t, mae_t, rmse_t = metrics(Y_test.iloc[:,1].values, Y_pred[:,1])
    return {"model":model_name, "best_params":best_params,
            "r2_c":r2_c,"mae_c":mae_c,"rmse_c":rmse_c,
            "r2_t":r2_t,"mae_t":mae_t,"rmse_t":rmse_t,
            "estimator":best}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="content/data.txt")
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--full-tune", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--save-csv", type=str, default="content/naval_results.csv")
    # Modify this line to ignore kernel arguments
    args = ap.parse_args([]) 
    np.random.seed(args.seed)
    df = load_data(args.data)
    X, Y, dropped = split_xy(df)
    print_block("DATA SHAPE")
    print(f"X: {X.shape},  Y: {Y.shape}")
    print(f"Dropped constant columns: {dropped}" if dropped else "No constant columns dropped.")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=args.test_size, random_state=args.seed)
    print_block("SINGLE-TARGET MODELS (per target)")
    rows = []
    for idx, tgt in enumerate(["gt_c_decay","gt_t_decay"]):
        ytr, yte = y_train.iloc[:,idx], y_test.iloc[:,idx]
        print(f"\nTarget: {tgt}")
        for mdl in ["RF","GBR"] + (["XGB"] if HAS_XGB else []):
            res = run_single_target(X_train, X_test, ytr, yte, model_name=mdl, full_tune=args.full_tune, n_jobs=args.n_jobs)
            if res is None: continue
            print(f"  {mdl}: R2={res['r2']:.4f} | MAE={res['mae']:.6f} | RMSE={res['rmse']:.6f}")
            if res["best_params"]: print(f"     best_params: {res['best_params']}")
            rows.append({"mode":"single","target":tgt,"model":mdl,"R2":res["r2"],"MAE":res["mae"],"RMSE":res["rmse"],"best_params":res["best_params"]})
    print_block("MULTI-OUTPUT MODEL (both targets together)")
    for mdl in ["RF","GBR"] + (["XGB"] if HAS_XGB else []):
        res = run_multi_output(X_train, X_test, y_train, y_test, model_name=mdl, full_tune=args.full_tune, n_jobs=args.n_jobs)
        if res is None: continue
        print(f"  {mdl}:")
        print(f"    kMc -> R2={res['r2_c']:.4f} | MAE={res['mae_c']:.6f} | RMSE={res['rmse_c']:.6f}")
        print(f"    kMt -> R2={res['r2_t']:.4f} | MAE={res['mae_t']:.6f} | RMSE={res['rmse_t']:.6f}")
        if res["best_params"]: print(f"     best_params: {res['best_params']}")
        rows.append({"mode":"multi","target":"gt_c_decay","model":mdl,"R2":res["r2_c"],"MAE":res["mae_c"],"RMSE":res["rmse_c"],"best_params":res["best_params"]})
        rows.append({"mode":"multi","target":"gt_t_decay","model":mdl,"R2":res["r2_t"],"MAE":res["mae_t"],"RMSE":res["rmse_t"],"best_params":res["best_params"]})
    out_df = pd.DataFrame(rows)
    out_path = os.path.abspath(args.save_csv)
    out_df.to_csv(out_path, index=False)
    print_block("SAVED")
    print(f"Results saved to: {out_path}")

if __name__ == "__main__":
    main()