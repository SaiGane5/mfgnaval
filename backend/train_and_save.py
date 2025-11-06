"""
Naval Propulsion Model - Inference Script
Usage: python predict.py --model app/model/model.joblib --input sample.csv
"""
import argparse
import numpy as np
import pandas as pd
from joblib import load

def create_engineered_features(X):
    """Create domain-specific engineered features (must match training)"""
    X_eng = X.copy()
    
    # Ratios and differences (physics-based)
    X_eng['torque_ratio'] = X_eng['gtt'] / (X_eng['ts'] + X_eng['tp'] + 1e-6)
    X_eng['temp_diff'] = X_eng['t2'] - X_eng['t1']
    X_eng['pressure_ratio'] = X_eng['p2'] / (X_eng['p1'] + 1e-6)
    X_eng['power_indicator'] = X_eng['gtt'] * X_eng['gtn']
    X_eng['efficiency_proxy'] = X_eng['v'] / (X_eng['mf'] + 1e-6)
    X_eng['prop_balance'] = np.abs(X_eng['ts'] - X_eng['tp'])
    
    # Temperature-pressure interactions
    X_eng['t_p_product'] = X_eng['t48'] * X_eng['p48']
    X_eng['compression_work'] = X_eng['temp_diff'] * X_eng['pressure_ratio']
    
    # Polynomial features
    poly_features = ['gtn', 'ggn', 't48', 'mf']
    for feat in poly_features:
        if feat in X_eng.columns:
            X_eng[f'{feat}_squared'] = X_eng[feat] ** 2
    
    return X_eng

def predict_from_model(model_path, input_data):
    """Load model and make predictions"""
    print(f"📦 Loading model from: {model_path}")
    
    # Load model package
    model_package = load(model_path)
    
    if isinstance(model_package, dict):
        pipeline = model_package['pipeline']
        selected_features = model_package['selected_features']
        performance = model_package['performance']
        
        print(f"✓ Model loaded successfully")
        print(f"  Performance: R² = {performance['avg_r2']:.4f}")
        print(f"  Features: {len(selected_features)}")
    else:
        # Legacy format (just pipeline)
        pipeline = model_package
        selected_features = None
        print(f"✓ Model loaded (legacy format)")
    
    # Prepare input
    if isinstance(input_data, str):
        # Load from file
        if input_data.endswith('.csv'):
            X = pd.read_csv(input_data)
        else:
            X = pd.read_csv(input_data, sep=r'\s+', header=None, engine='python')
            # Assign column names
            expected_cols = ['lp','v','gtt','gtn','ggn','ts','tp','t48',
                           't1','t2','p48','p1','p2','pexh','tic','mf']
            if len(X.columns) == len(expected_cols):
                X.columns = expected_cols
    elif isinstance(input_data, pd.DataFrame):
        X = input_data.copy()
    else:
        raise ValueError("input_data must be filepath or DataFrame")
    
    print(f"\n📊 Input data: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Engineer features
    X_eng = create_engineered_features(X)
    
    # Select features if model has feature list
    if selected_features:
        missing = [f for f in selected_features if f not in X_eng.columns]
        if missing:
            print(f"⚠️  Warning: Missing features: {missing}")
        X_final = X_eng[[f for f in selected_features if f in X_eng.columns]]
    else:
        X_final = X_eng
    
    print(f"   Features after engineering: {X_final.shape[1]}")
    
    # Predict
    print(f"\n🔮 Making predictions...")
    predictions = pipeline.predict(X_final)
    
    # Format results
    results = pd.DataFrame({
        'gt_compressor_decay': predictions[:, 0],
        'gt_turbine_decay': predictions[:, 1]
    })
    
    # Add input features for reference
    results = pd.concat([X.reset_index(drop=True), results], axis=1)
    
    print(f"✅ Predictions complete!")
    print(f"\n📈 Summary Statistics:")
    print(results[['gt_compressor_decay', 'gt_turbine_decay']].describe())
    
    return results

def main():
    ap = argparse.ArgumentParser(description='Predict naval propulsion decay coefficients')
    ap.add_argument('--model', type=str, required=True, help='Path to trained model')
    ap.add_argument('--input', type=str, required=True, help='Input data file (CSV or whitespace-delimited)')
    ap.add_argument('--output', type=str, default=None, help='Output CSV file (optional)')
    args = ap.parse_args()
    
    print("="*70)
    print("🚢 NAVAL PROPULSION DECAY PREDICTION")
    print("="*70)
    
    # Make predictions
    results = predict_from_model(args.model, args.input)
    
    # Save if output specified
    if args.output:
        results.to_csv(args.output, index=False)
        print(f"\n💾 Results saved to: {args.output}")
    else:
        print(f"\n📋 First 5 predictions:")
        print(results.head())
    
    print(f"\n✅ Done!")

if __name__ == '__main__':
    main()