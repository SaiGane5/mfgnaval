"""
Enhanced Naval Propulsion Plant - Advanced Feature Engineering & Selection
Features: Polynomial features, interaction terms, PCA, feature importance, recursive elimination
"""
import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from joblib import dump

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import (StandardScaler, RobustScaler, PowerTransformer, 
                                    QuantileTransformer, MinMaxScaler, PolynomialFeatures)
from sklearn.decomposition import PCA
from sklearn.feature_selection import (SelectKBest, f_regression, mutual_info_regression,
                                        RFE, VarianceThreshold, SelectFromModel)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                               ExtraTreesRegressor, BaggingRegressor)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge, Lasso

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

DEFAULT_FEATURE_NAMES = [
    'lp','v','gtt','gtn','ggn','ts','tp','t48','t1','t2','p48','p1','p2','pexh','tic','mf',
    'gt_c_decay','gt_t_decay'
]
RANDOM_STATE = 42

# Statistical tests
def test_normality(data, name="Data"):
    """Shapiro-Wilk: H0: data ~ Normal"""
    stat, p = stats.shapiro(data[:5000] if len(data) > 5000 else data)
    return {'name': name, 'test': 'Shapiro-Wilk', 'statistic': stat, 
            'p_value': p, 'is_normal': p > 0.05}

def test_zero_mean(data, name="Residuals"):
    """One-sample t-test: H0: μ = 0"""
    stat, p = stats.ttest_1samp(data, 0)
    return {'name': name, 'test': 't-test', 'statistic': stat, 
            'p_value': p, 'mean': np.mean(data), 'unbiased': p > 0.05}

def test_homoscedasticity(y_pred, residuals):
    """Levene: H0: equal variances across prediction ranges"""
    indices = np.argsort(y_pred)
    n = len(y_pred) // 5
    groups = [residuals[indices[i*n:(i+1)*n]] for i in range(4)]
    groups.append(residuals[indices[4*n:]])
    stat, p = stats.levene(*groups)
    return {'test': 'Levene', 'statistic': stat, 'p_value': p, 'homoscedastic': p > 0.05}

# Core functions
def load_data(path, feature_names=DEFAULT_FEATURE_NAMES):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path, sep=r'\s+', header=None, engine='python')
    df.columns = feature_names
    return df

def split_xy(df):
    const_cols = [c for c in df.columns if df[c].nunique() <= 1]
    const_cols = [c for c in const_cols if c not in ("gt_c_decay","gt_t_decay")]
    df = df.drop(columns=const_cols) if const_cols else df
    X = df.drop(columns=["gt_c_decay","gt_t_decay"])
    y = df[["gt_c_decay","gt_t_decay"]]
    return X, y

def analyze_feature_correlations(X, y, output_dir='results'):
    """Analyze and visualize feature correlations"""
    print("\n📊 Feature Correlation Analysis:")
    
    # Combine for full correlation matrix
    df_full = pd.concat([X, y], axis=1)
    corr_matrix = df_full.corr()
    
    # Target correlations
    target_corrs = corr_matrix[['gt_c_decay', 'gt_t_decay']].drop(['gt_c_decay', 'gt_t_decay'])
    target_corrs['abs_mean'] = target_corrs.abs().mean(axis=1)
    target_corrs = target_corrs.sort_values('abs_mean', ascending=False)
    
    print("\n   Top 10 Features by Correlation with Targets:")
    for feat, row in target_corrs.head(10).iterrows():
        print(f"   {feat:10s}: C={row['gt_c_decay']:+.3f}, T={row['gt_t_decay']:+.3f}")
    
    # Multicollinearity check (VIF approximation)
    print("\n   Multicollinearity Check (|r| > 0.9):")
    high_corr_pairs = []
    for i in range(len(X.columns)):
        for j in range(i+1, len(X.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.9:
                high_corr_pairs.append((X.columns[i], X.columns[j], corr_val))
    
    if high_corr_pairs:
        for feat1, feat2, corr in high_corr_pairs:
            print(f"   {feat1} <-> {feat2}: {corr:.3f}")
    else:
        print("   ✓ No severe multicollinearity detected")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Full correlation heatmap
    ax = axes[0]
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=False, 
                square=True, ax=ax, vmin=-1, vmax=1, cbar_kws={'shrink': 0.8})
    ax.set_title('Full Correlation Matrix', fontweight='bold', fontsize=12)
    
    # Target correlations
    ax = axes[1]
    target_corrs_plot = target_corrs.drop('abs_mean', axis=1).head(15)
    target_corrs_plot.plot(kind='barh', ax=ax, width=0.8)
    ax.set_xlabel('Correlation Coefficient')
    ax.set_title('Top 15 Features - Correlation with Targets', fontweight='bold', fontsize=12)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.legend(['GT Compressor', 'GT Turbine'])
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return target_corrs

def create_engineered_features(X):
    """Create domain-specific engineered features"""
    X_eng = X.copy()
    
    # Ratios and differences (physics-based)
    # Only compute engineered features when required base columns exist.
    if {'gtt', 'ts', 'tp'}.issubset(X_eng.columns):
        denom = (X_eng['ts'] + X_eng['tp'] + 1e-6)
        X_eng['torque_ratio'] = X_eng['gtt'] / denom

    if {'t2', 't1'}.issubset(X_eng.columns):
        X_eng['temp_diff'] = X_eng['t2'] - X_eng['t1']  # Compression heating

    if {'p2', 'p1'}.issubset(X_eng.columns):
        X_eng['pressure_ratio'] = X_eng['p2'] / (X_eng['p1'] + 1e-6)

    if {'gtt', 'gtn'}.issubset(X_eng.columns):
        X_eng['power_indicator'] = X_eng['gtt'] * X_eng['gtn']  # Torque × RPM

    if {'v', 'mf'}.issubset(X_eng.columns):
        X_eng['efficiency_proxy'] = X_eng['v'] / (X_eng['mf'] + 1e-6)  # Speed per fuel

    if {'ts', 'tp'}.issubset(X_eng.columns):
        X_eng['prop_balance'] = np.abs(X_eng['ts'] - X_eng['tp'])  # Propeller balance
    
    # Temperature-pressure interactions (guard columns)
    if {'t48', 'p48'}.issubset(X_eng.columns):
        X_eng['t_p_product'] = X_eng['t48'] * X_eng['p48']

    # compression_work depends on temp_diff and pressure_ratio which may be engineered above
    if {'temp_diff', 'pressure_ratio'}.issubset(X_eng.columns):
        X_eng['compression_work'] = X_eng['temp_diff'] * X_eng['pressure_ratio']
    
    # Polynomial features for key variables (degree 2)
    poly_features = ['gtn', 'ggn', 't48', 'mf']
    for feat in poly_features:
        if feat in X_eng.columns:
            X_eng[f'{feat}_squared'] = X_eng[feat] ** 2
    
    print(f"\n   ✓ Engineered {len(X_eng.columns) - len(X.columns)} new features")
    return X_eng

def feature_importance_analysis(X_train, y_train, X_test, y_test, original_features, output_dir='results'):
    """Comprehensive feature importance using multiple methods"""
    print("\n🔍 Feature Importance Analysis:")
    
    importance_scores = pd.DataFrame(index=X_train.columns)
    
    # Method 1: Random Forest
    rf = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    rf_r2 = np.mean([r2_score(y_test.iloc[:, i], y_pred[:, i]) for i in range(2)])
    
    # Average importance across targets
    importances = np.mean([est.feature_importances_ for est in rf.estimators_], axis=0)
    importance_scores['RF_importance'] = importances
    
    # Method 2: Gradient Boosting
    gbr = GradientBoostingRegressor(n_estimators=300, random_state=RANDOM_STATE)
    gbr_c = gbr.fit(X_train, y_train.iloc[:, 0])
    gbr_t = GradientBoostingRegressor(n_estimators=300, random_state=RANDOM_STATE).fit(X_train, y_train.iloc[:, 1])
    importance_scores['GBR_importance'] = (gbr_c.feature_importances_ + gbr_t.feature_importances_) / 2
    
    # Method 3: Mutual Information
    mi_c = mutual_info_regression(X_train, y_train.iloc[:, 0], random_state=RANDOM_STATE)
    mi_t = mutual_info_regression(X_train, y_train.iloc[:, 1], random_state=RANDOM_STATE)
    importance_scores['MI_score'] = (mi_c + mi_t) / 2
    
    # Method 4: F-statistic
    f_scores_c, _ = f_regression(X_train, y_train.iloc[:, 0])
    f_scores_t, _ = f_regression(X_train, y_train.iloc[:, 1])
    importance_scores['F_statistic'] = (f_scores_c + f_scores_t) / 2
    
    # Normalize and average
    for col in importance_scores.columns:
        importance_scores[f'{col}_norm'] = (importance_scores[col] - importance_scores[col].min()) / \
                                           (importance_scores[col].max() - importance_scores[col].min() + 1e-10)
    
    norm_cols = [c for c in importance_scores.columns if '_norm' in c]
    importance_scores['avg_importance'] = importance_scores[norm_cols].mean(axis=1)
    importance_scores = importance_scores.sort_values('avg_importance', ascending=False)
    
    # Mark original vs engineered
    importance_scores['type'] = ['Original' if f in original_features else 'Engineered' 
                                  for f in importance_scores.index]
    
    print(f"\n   Top 15 Most Important Features:")
    for idx, (feat, row) in enumerate(importance_scores.head(15).iterrows(), 1):
        feat_type = "🔧" if row['type'] == 'Engineered' else "📊"
        print(f"   {idx:2d}. {feat_type} {feat:25s}: {row['avg_importance']:.4f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Top features by method
    ax = axes[0, 0]
    top_features = importance_scores.head(20)
    methods = ['RF_importance', 'GBR_importance', 'MI_score']
    top_features[methods].plot(kind='barh', ax=ax)
    ax.set_xlabel('Importance Score')
    ax.set_title('Top 20 Features - Multiple Methods', fontweight='bold')
    ax.legend(['Random Forest', 'Gradient Boosting', 'Mutual Information'])
    ax.invert_yaxis()
    
    # Average importance
    ax = axes[0, 1]
    top_avg = importance_scores.head(20)
    colors = ['steelblue' if t == 'Original' else 'coral' for t in top_avg['type']]
    ax.barh(range(len(top_avg)), top_avg['avg_importance'], color=colors)
    ax.set_yticks(range(len(top_avg)))
    ax.set_yticklabels(top_avg.index)
    ax.set_xlabel('Average Normalized Importance')
    ax.set_title('Top 20 Features - Average Importance', fontweight='bold')
    ax.invert_yaxis()
    
    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='steelblue', label='Original'),
                      Patch(facecolor='coral', label='Engineered')]
    ax.legend(handles=legend_elements)
    
    # Cumulative importance
    ax = axes[1, 0]
    cumsum = importance_scores['avg_importance'].cumsum()
    cumsum_pct = cumsum / cumsum.iloc[-1] * 100
    ax.plot(range(len(cumsum_pct)), cumsum_pct, linewidth=2)
    ax.axhline(y=90, color='red', linestyle='--', label='90% threshold')
    ax.axhline(y=95, color='orange', linestyle='--', label='95% threshold')
    ax.set_xlabel('Number of Features')
    ax.set_ylabel('Cumulative Importance (%)')
    ax.set_title('Cumulative Feature Importance', fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend()
    
    # Feature type comparison
    ax = axes[1, 1]
    type_importance = importance_scores.groupby('type')['avg_importance'].agg(['sum', 'mean', 'count'])
    type_importance.plot(kind='bar', ax=ax)
    ax.set_ylabel('Importance Score')
    ax.set_title('Original vs Engineered Features', fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(['Total', 'Mean', 'Count'])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return importance_scores, rf_r2

def feature_selection_comparison(X_train, y_train, X_test, y_test, output_dir='results'):
    """Compare different feature selection methods"""
    print("\n🎯 Feature Selection Method Comparison:")
    
    results = []
    base_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1))
    
    # Baseline: All features
    base_model.fit(X_train, y_train)
    y_pred = base_model.predict(X_test)
    baseline_r2 = np.mean([r2_score(y_test.iloc[:, i], y_pred[:, i]) for i in range(2)])
    results.append({'method': 'All Features', 'n_features': X_train.shape[1], 'r2': baseline_r2})
    print(f"\n   Baseline (All {X_train.shape[1]} features): R² = {baseline_r2:.4f}")
    
    # Method 1: Variance Threshold
    selector = VarianceThreshold(threshold=0.01)
    X_train_sel = selector.fit_transform(X_train)
    X_test_sel = selector.transform(X_test)
    base_model.fit(X_train_sel, y_train)
    y_pred = base_model.predict(X_test_sel)
    r2 = np.mean([r2_score(y_test.iloc[:, i], y_pred[:, i]) for i in range(2)])
    results.append({'method': 'Variance Threshold', 'n_features': X_train_sel.shape[1], 'r2': r2})
    print(f"   Variance Threshold ({X_train_sel.shape[1]} features): R² = {r2:.4f}")
    
    # Method 2: SelectKBest with different k values
    for k in [10, 15, 20, 25]:
        if k > X_train.shape[1]:
            continue
        selector = SelectKBest(f_regression, k=k)
        X_train_sel = selector.fit_transform(X_train, y_train.iloc[:, 0])  # Use first target
        X_test_sel = selector.transform(X_test)
        base_model.fit(X_train_sel, y_train)
        y_pred = base_model.predict(X_test_sel)
        r2 = np.mean([r2_score(y_test.iloc[:, i], y_pred[:, i]) for i in range(2)])
        results.append({'method': f'SelectKBest (k={k})', 'n_features': k, 'r2': r2})
        print(f"   SelectKBest k={k}: R² = {r2:.4f}")
    
    # Method 3: RFE
    estimator = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    for n_feat in [15, 20]:
        if n_feat > X_train.shape[1]:
            continue
        selector = RFE(estimator, n_features_to_select=n_feat, step=5)
        X_train_sel = selector.fit_transform(X_train, y_train.iloc[:, 0])
        X_test_sel = selector.transform(X_test)
        base_model.fit(X_train_sel, y_train)
        y_pred = base_model.predict(X_test_sel)
        r2 = np.mean([r2_score(y_test.iloc[:, i], y_pred[:, i]) for i in range(2)])
        results.append({'method': f'RFE (n={n_feat})', 'n_features': n_feat, 'r2': r2})
        print(f"   RFE n={n_feat}: R² = {r2:.4f}")
    
    # Method 4: L1-based (Lasso)
    lasso = Lasso(alpha=0.001, random_state=RANDOM_STATE)
    selector = SelectFromModel(lasso, prefit=False)
    X_train_sel = selector.fit_transform(X_train, y_train.iloc[:, 0])
    X_test_sel = selector.transform(X_test)
    if X_train_sel.shape[1] > 0:
        base_model.fit(X_train_sel, y_train)
        y_pred = base_model.predict(X_test_sel)
        r2 = np.mean([r2_score(y_test.iloc[:, i], y_pred[:, i]) for i in range(2)])
        results.append({'method': 'L1-Lasso', 'n_features': X_train_sel.shape[1], 'r2': r2})
        print(f"   L1-Lasso ({X_train_sel.shape[1]} features): R² = {r2:.4f}")
    
    # Method 5: Tree-based (RF)
    rf = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    selector = SelectFromModel(rf, prefit=False, threshold='median')
    X_train_sel = selector.fit_transform(X_train, y_train.iloc[:, 0])
    X_test_sel = selector.transform(X_test)
    base_model.fit(X_train_sel, y_train)
    y_pred = base_model.predict(X_test_sel)
    r2 = np.mean([r2_score(y_test.iloc[:, i], y_pred[:, i]) for i in range(2)])
    results.append({'method': 'Tree-based (median)', 'n_features': X_train_sel.shape[1], 'r2': r2})
    print(f"   Tree-based ({X_train_sel.shape[1]} features): R² = {r2:.4f}")
    
    # Visualization
    results_df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # R² comparison
    ax = axes[0]
    colors = ['green' if r > baseline_r2 else 'red' for r in results_df['r2']]
    ax.barh(results_df['method'], results_df['r2'], color=colors, alpha=0.7)
    ax.axvline(baseline_r2, color='blue', linestyle='--', linewidth=2, label='Baseline')
    ax.set_xlabel('R² Score')
    ax.set_title('Feature Selection Performance', fontweight='bold')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    # Features vs Performance
    ax = axes[1]
    ax.scatter(results_df['n_features'], results_df['r2'], s=100, alpha=0.7)
    for _, row in results_df.iterrows():
        ax.annotate(row['method'], (row['n_features'], row['r2']), 
                   fontsize=8, alpha=0.7, ha='right')
    ax.axhline(baseline_r2, color='blue', linestyle='--', label='Baseline R²')
    ax.set_xlabel('Number of Features')
    ax.set_ylabel('R² Score')
    ax.set_title('Feature Count vs Performance', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_selection_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return results_df

def create_preprocessing_strategies(X):
    """Create diverse preprocessing pipelines"""
    num_cols = X.columns.tolist()
    
    strategies = {
        'standard': ColumnTransformer([('scale', StandardScaler(), num_cols)], remainder='drop'),
        'robust': ColumnTransformer([('scale', RobustScaler(), num_cols)], remainder='drop'),
        'minmax': ColumnTransformer([('scale', MinMaxScaler(), num_cols)], remainder='drop'),
        'power': ColumnTransformer([('scale', PowerTransformer(method='yeo-johnson'), num_cols)], remainder='drop'),
        'quantile': ColumnTransformer([('scale', QuantileTransformer(output_distribution='normal'), num_cols)], remainder='drop'),
    }
    
    return strategies

def train_robust_models(data_path, output_dir='results', save_model=True, n_jobs=-1):
    """Enhanced training with feature engineering and selection"""
    print("="*90)
    print("🚢 NAVAL PROPULSION - ADVANCED FEATURE ENGINEERING & SELECTION")
    print("="*90)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load
    df = load_data(data_path)
    X, y = split_xy(df)
    original_features = X.columns.tolist()
    
    print(f"\n📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE
    )
    
    # Step 1: Correlation Analysis
    analyze_feature_correlations(X_train, y_train, output_dir)
    
    # Step 2: Feature Engineering
    print(f"\n{'─'*90}")
    print("🔧 Feature Engineering:")
    X_train_eng = create_engineered_features(X_train)
    X_test_eng = create_engineered_features(X_test)
    print(f"   Total features: {len(original_features)} → {X_train_eng.shape[1]}")
    
    # Step 3: Feature Importance
    print(f"\n{'─'*90}")
    importance_scores, baseline_r2 = feature_importance_analysis(
        X_train_eng, y_train, X_test_eng, y_test, original_features, output_dir
    )
    
    # Step 4: Feature Selection Comparison
    print(f"\n{'─'*90}")
    selection_results = feature_selection_comparison(
        X_train_eng, y_train, X_test_eng, y_test, output_dir
    )
    
    # Step 5: Select best features based on importance
    print(f"\n{'─'*90}")
    print("🎯 Selecting Top Features for Final Model:")
    
    # Take top features that cover 95% cumulative importance
    cumsum = importance_scores['avg_importance'].cumsum()
    cumsum_pct = cumsum / cumsum.iloc[-1]
    n_features_95 = (cumsum_pct <= 0.95).sum()
    top_features = importance_scores.head(max(n_features_95, 20)).index.tolist()
    
    print(f"   Selected {len(top_features)} features (95% cumulative importance)")
    
    X_train_final = X_train_eng[top_features]
    X_test_final = X_test_eng[top_features]
    
    # Step 6: Train with multiple preprocessing strategies
    print(f"\n{'─'*90}")
    print("⚙️  Training Final Models:")
    
    preprocessors = create_preprocessing_strategies(X_train_final)
    
    models = {
        'RF': RandomForestRegressor(n_estimators=600, max_depth=None, min_samples_split=2,
                                    n_jobs=n_jobs, random_state=RANDOM_STATE),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=600, max_depth=None, min_samples_split=2,
                                          n_jobs=n_jobs, random_state=RANDOM_STATE),
        'GBR': GradientBoostingRegressor(n_estimators=600, learning_rate=0.05, max_depth=4,
                                          subsample=0.8, random_state=RANDOM_STATE),
    }
    
    if HAS_XGB:
        models['XGB'] = XGBRegressor(n_estimators=600, learning_rate=0.07, max_depth=5,
                                     subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_STATE)
    
    results = []
    best_r2 = -np.inf
    best_pipeline = None
    best_config = None
    
    for prep_name, preprocessor in preprocessors.items():
        print(f"\n   Preprocessing: {prep_name.upper()}")
        
        for model_name, base_model in models.items():
            model = MultiOutputRegressor(base_model)
            pipeline = Pipeline([('prep', preprocessor), ('model', model)])
            
            pipeline.fit(X_train_final, y_train)
            y_pred = pipeline.predict(X_test_final)
            
            for idx, target in enumerate(y.columns):
                y_true = y_test.iloc[:, idx].values
                y_p = y_pred[:, idx]
                
                r2 = r2_score(y_true, y_p)
                mae = mean_absolute_error(y_true, y_p)
                rmse = np.sqrt(mean_squared_error(y_true, y_p))
                
                results.append({
                    'preprocessing': prep_name,
                    'model': model_name,
                    'target': target,
                    'r2': r2,
                    'mae': mae,
                    'rmse': rmse,
                })
            
            avg_r2 = np.mean([r2_score(y_test.iloc[:, i], y_pred[:, i]) for i in range(2)])
            print(f"      {model_name:12s}: R² = {avg_r2:.4f}")
            
            if avg_r2 > best_r2:
                best_r2 = avg_r2
                best_pipeline = pipeline
                best_config = (prep_name, model_name)
    
    # Final Results
    print(f"\n{'='*90}")
    print(f"✅ BEST MODEL: {best_config[0]} + {best_config[1]}")
    print(f"   Average R²: {best_r2:.6f}")
    print(f"   Features Used: {len(top_features)}/{X_train_eng.shape[1]}")
    print(f"{'='*90}")
    
    # Validate best model
    y_pred_best = best_pipeline.predict(X_test_final)
    for idx, target in enumerate(y.columns):
        y_true = y_test.iloc[:, idx].values
        y_p = y_pred_best[:, idx]
        residuals = y_true - y_p
        
        r2 = r2_score(y_true, y_p)
        mae = mean_absolute_error(y_true, y_p)
        rmse = np.sqrt(mean_squared_error(y_true, y_p))
        
        print(f"\n📌 {target}:")
        print(f"   R² = {r2:.6f} | MAE = {mae:.8f} | RMSE = {rmse:.8f}")
        
        norm = test_normality(residuals)
        bias = test_zero_mean(residuals)
        homo = test_homoscedasticity(y_p, residuals)
        
        print(f"   🔬 Normality: p={norm['p_value']:.4f} → {'✓' if norm['is_normal'] else '✗'}")
        print(f"   🔬 Bias: μ={bias['mean']:.2e}, p={bias['p_value']:.4f} → {'✓' if bias['unbiased'] else '✗'}")
        print(f"   🔬 Homoscedasticity: p={homo['p_value']:.4f} → {'✓' if homo['homoscedastic'] else '✗'}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{output_dir}/results.csv', index=False)
    
    # Save feature importance
    importance_scores.to_csv(f'{output_dir}/feature_importance.csv')
    
    # Save selected features
    with open(f'{output_dir}/selected_features.txt', 'w') as f:
        f.write(f"Total features: {len(top_features)}\n")
        f.write(f"Selected from: {X_train_eng.shape[1]} engineered features\n\n")
        f.write("Selected Features:\n")
        for i, feat in enumerate(top_features, 1):
            imp = importance_scores.loc[feat, 'avg_importance']
            feat_type = importance_scores.loc[feat, 'type']
            f.write(f"{i:3d}. {feat:25s} (importance: {imp:.4f}, type: {feat_type})\n")
    
    print(f"\n💾 Results saved to '{output_dir}/'")
    
    # Save best model
    if save_model:
        model_path = f'{output_dir}/best_model.joblib'
        model_info = {
            'pipeline': best_pipeline,
            'features': top_features,
            'config': best_config,
            'performance': {
                'r2_compressor': r2_score(y_test.iloc[:, 0], y_pred_best[:, 0]),
                'r2_turbine': r2_score(y_test.iloc[:, 1], y_pred_best[:, 1]),
                'avg_r2': best_r2
            }
        }
        dump(model_info, model_path)
        print(f"💾 Model saved: {model_path}")
    
    # Generate summary plots
    plot_summary(results_df, importance_scores, selection_results, output_dir)
    
    print(f"\n✅ Training complete!")
    return results_df, best_pipeline, top_features

def plot_summary(results_df, importance_scores, selection_results, output_dir):
    """Generate summary visualization"""
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Model performance heatmap
    ax = fig.add_subplot(gs[0, :2])
    pivot = results_df.pivot_table(values='r2', index='model', columns='preprocessing')
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn', ax=ax, 
                vmin=0.90, vmax=1.0, cbar_kws={'label': 'R² Score'})
    ax.set_title('Model Performance Grid (R²)', fontweight='bold', fontsize=12)
    
    # 2. Best model per target
    ax = fig.add_subplot(gs[0, 2])
    target_best = results_df.groupby('target')['r2'].max().sort_values()
    target_best.plot(kind='barh', ax=ax, color=['steelblue', 'coral'])
    ax.set_xlabel('R² Score')
    ax.set_title('Best R² per Target', fontweight='bold', fontsize=11)
    ax.axvline(0.95, color='red', linestyle='--', linewidth=1, alpha=0.7)
    
    # 3. Top features
    ax = fig.add_subplot(gs[1, :2])
    top_15 = importance_scores.head(15)
    colors_feat = ['steelblue' if t == 'Original' else 'coral' for t in top_15['type']]
    ax.barh(range(len(top_15)), top_15['avg_importance'], color=colors_feat)
    ax.set_yticks(range(len(top_15)))
    ax.set_yticklabels(top_15.index, fontsize=9)
    ax.set_xlabel('Importance Score')
    ax.set_title('Top 15 Features', fontweight='bold', fontsize=11)
    ax.invert_yaxis()
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='steelblue', label='Original'),
                      Patch(facecolor='coral', label='Engineered')]
    ax.legend(handles=legend_elements, loc='lower right')
    
    # 4. Feature selection comparison
    ax = fig.add_subplot(gs[1, 2])
    selection_results_sorted = selection_results.sort_values('r2', ascending=True)
    colors_sel = ['green' if r > selection_results[selection_results['method']=='All Features']['r2'].values[0] 
                  else 'red' for r in selection_results_sorted['r2']]
    ax.barh(selection_results_sorted['method'], selection_results_sorted['r2'], 
            color=colors_sel, alpha=0.7)
    ax.set_xlabel('R² Score')
    ax.set_title('Feature Selection Methods', fontweight='bold', fontsize=11)
    ax.tick_params(axis='y', labelsize=8)
    
    # 5. MAE distribution
    ax = fig.add_subplot(gs[2, 0])
    results_df.boxplot(column='mae', by='model', ax=ax)
    ax.set_ylabel('MAE')
    ax.set_title('MAE Distribution by Model', fontweight='bold', fontsize=11)
    plt.sca(ax)
    plt.xticks(rotation=45, fontsize=9)
    
    # 6. RMSE distribution
    ax = fig.add_subplot(gs[2, 1])
    results_df.boxplot(column='rmse', by='model', ax=ax)
    ax.set_ylabel('RMSE')
    ax.set_title('RMSE Distribution by Model', fontweight='bold', fontsize=11)
    plt.sca(ax)
    plt.xticks(rotation=45, fontsize=9)
    
    # 7. Preprocessing impact
    ax = fig.add_subplot(gs[2, 2])
    prep_impact = results_df.groupby('preprocessing')['r2'].agg(['mean', 'std'])
    prep_impact['mean'].plot(kind='bar', ax=ax, yerr=prep_impact['std'], 
                             capsize=4, color='mediumpurple', alpha=0.7)
    ax.set_ylabel('R² Score')
    ax.set_title('Preprocessing Impact', fontweight='bold', fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Naval Propulsion Model - Comprehensive Analysis', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.savefig(f'{output_dir}/summary_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Summary dashboard saved")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='content/data.txt')
    ap.add_argument('--output-dir', default='results')
    ap.add_argument('--n-jobs', type=int, default=-1)
    args = ap.parse_args()
    
    train_robust_models(args.data, args.output_dir, n_jobs=args.n_jobs)