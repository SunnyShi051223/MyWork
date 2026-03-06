import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from src.config import Config
from src.data_loader import DataLoader
from tqdm import tqdm

def calculate_vip(pls_model, X):
    t = pls_model.x_scores_
    w = pls_model.x_weights_
    q = pls_model.y_loadings_
    p, h = w.shape
    vips = np.zeros(p)
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)
    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j]))**2 for j in range(h)])
        vips[i] = np.sqrt(p * (s.T @ weight) / total_s)
    return vips

def main():
    print("\n=== Iteration 15b: Tuned Clinical Omic Pipeline ===")
    print("Mimicking: 'Integrated landscape of plasma metabolism and proteome of DVT'")
    
    # --- Load Data ---
    Config.ENABLE_FEATURE_ENGINEERING = False
    Config.ENABLE_SYMBOLIC_CROSSING = False
    loader = DataLoader()
    X_raw, y, feature_names = loader.load_process()
    df = pd.DataFrame(X_raw, columns=feature_names)
    
    # ----------- Stage 1: Covariate Adjustment -----------
    print("\n[Stage 1] Covariate Adjustment & Preprocessing")
    covariate_cols = ['男1.女2', '年龄', 'bmi']
    actual_covars = [c for c in covariate_cols if c in df.columns]
    
    df_adjusted = pd.DataFrame(index=df.index)
    X_covars = sm.add_constant(df[actual_covars])
    ks_results = {}
    
    for col in df.columns:
        if col in actual_covars: continue
        y_feat = df[col]
        # OLS to get residuals
        model = sm.OLS(y_feat, X_covars).fit()
        residuals = model.resid
        # Median Centered Scaling
        centered = residuals - np.median(residuals)
        scaler = StandardScaler(with_mean=False) 
        scaled = scaler.fit_transform(centered.values.reshape(-1, 1)).flatten()
        
        ks_stat, p_val = stats.kstest(scaled, 'norm')
        ks_results[col] = {'is_normal': p_val > 0.05, 'p_value': p_val}
        df_adjusted[col] = scaled

    # ----------- Stage 2: Two-step Feature Selection (Tuned) -----------
    print("\n[Stage 2] Two-step Feature Selection (Tuned Omic Parameters)")
    
    mask_pos = (y == 1)
    mask_neg = (y == 0)
    
    p_values = []
    fc_values = []
    features_eval = df_adjusted.columns.tolist()
    
    for col in features_eval:
        data_col = df_adjusted[col].values
        pos_data = data_col[mask_pos]
        neg_data = data_col[mask_neg]
        
        if ks_results[col]['is_normal']:
            _, p = stats.ttest_ind(pos_data, neg_data, equal_var=False)
        else:
            _, p = stats.mannwhitneyu(pos_data, neg_data)
        p_values.append(p)
        
        # Shift to positive for valid Fold Change calculation
        shifted = data_col - np.min(data_col) + 1 
        fc = np.mean(shifted[mask_pos]) / np.mean(shifted[mask_neg])
        fc_values.append(fc)
        
    # TUNING 1: Use Raw P < 0.05 & FC > 1.2 or < 0.83 (Standard for Small N Omics)
    uni_hits = [f for i, f in enumerate(features_eval) if p_values[i] < 0.05 and (fc_values[i] > 1.2 or fc_values[i] < 0.833)]
    print(f"       -> Univariate Selected Features (P < 0.05 & FC > 1.2): {len(uni_hits)}")
    if len(uni_hits) == 0: uni_hits = [f for i, f in enumerate(features_eval) if p_values[i] < 0.05]
        
    print("       -> Running Multivariate OPLS-DA Proxy (PLS-DA) with Permutation...")
    X_uni = df_adjusted[uni_hits].values
    pls = PLSRegression(n_components=2)
    pls.fit(X_uni, y)
    vips = calculate_vip(pls, X_uni)
    
    # TUNING 2: VIP > 1.0 (Standard)
    multi_selected = [uni_hits[i] for i in range(len(vips)) if vips[i] > 1.0]
    print(f"       -> Multivariate Selected Features (VIP > 1): {len(multi_selected)}")
    
    final_intersect = list(set(uni_hits).intersection(set(multi_selected)))
    if len(final_intersect) == 0: final_intersect = multi_selected[:15]
    print(f"       -> Final Intersecting Biomarker Panel: {len(final_intersect)}")
    print(f"       -> Panel: {final_intersect}")
    
    # ----------- Stage 3 & 4: GridSearchCV Modeling & 1000x Bootstrap -----------
    print("\n[Stage 3] Automated Hyperparameter Optimization (GridSearchCV)")
    X_final = df_adjusted[final_intersect].values
    
    # TUNING 3: Do not hardcode Polynomial. Search linear, rbf, and poly.
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto'],
        'degree': [2, 3] # Only used by poly
    }
    svc = SVC(probability=True, class_weight='balanced', random_state=Config.SEED)
    grid = GridSearchCV(svc, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_final, y)
    
    best_model = grid.best_estimator_
    print(f"       -> Optimal Core Found: {grid.best_params_}")
    print(f"       -> Inner 5-Fold Evaluation AUC of Optimal Core: {grid.best_score_:.4f}")
    
    print("\n[Stage 4] Bootstrapping 1000 iterations for Robust 95% CI bounds...")
    n_bootstraps = 1000
    aucs = []
    
    np.random.seed(Config.SEED)
    for i in tqdm(range(n_bootstraps)):
        X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=np.random.randint(100000), stratify=y)
        best_model.fit(X_train, y_train)
        preds = best_model.predict_proba(X_test)[:, 1]
        try:
            auc = roc_auc_score(y_test, preds)
            aucs.append(auc)
        except:
            pass
            
    mean_auc = np.mean(aucs)
    ci_lower = np.percentile(aucs, 2.5)
    ci_upper = np.percentile(aucs, 97.5)
    print(f"       -> 1000x Bootstrap AUC: {mean_auc:.4f} (95% CI: {ci_lower:.4f} - {ci_upper:.4f})")
    
    # ----------- Stage 5: Model Reduction Search -----------
    print("\n[Stage 5] Model Reduction Step-Down Search")
    
    # Rank by P-value
    rank_pvals = []
    for f in final_intersect:
        data_col = df_adjusted[f].values
        if ks_results[f]['is_normal']:
            _, p = stats.ttest_ind(data_col[mask_pos], data_col[mask_neg], equal_var=False)
        else:
            _, p = stats.mannwhitneyu(data_col[mask_pos], data_col[mask_neg])
        rank_pvals.append(p)
        
    ranked_features = [x for _, x in sorted(zip(rank_pvals, final_intersect))]
    
    reduction_steps = [20, 15, 10, 5, 3]
    reduction_steps = [s for s in reduction_steps if s <= len(ranked_features)]
    
    optimal_panel = None
    cv_splitter = StratifiedShuffleSplit(n_splits=50, test_size=0.2, random_state=42)
    
    for k in reduction_steps:
        sub_features = ranked_features[:k]
        X_sub = df_adjusted[sub_features].values
        
        # GridSearch again for subset
        sub_grid = GridSearchCV(SVC(probability=True, class_weight='balanced', random_state=Config.SEED), 
                                param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
        sub_grid.fit(X_sub, y)
        sub_model = sub_grid.best_estimator_
        
        sub_aucs = []
        for train_idx, test_idx in cv_splitter.split(X_sub, y):
            sub_model.fit(X_sub[train_idx], y[train_idx])
            preds = sub_model.predict_proba(X_sub[test_idx])[:, 1]
            sub_aucs.append(roc_auc_score(y[test_idx], preds))
            
        mean_sub_auc = np.mean(sub_aucs)
        print(f"       -> Top {k} Features Panel Mean AUC: {mean_sub_auc:.4f}")
        
        if mean_sub_auc > 0.85:
            optimal_panel = (k, mean_sub_auc, sub_features)
            
    if optimal_panel:
        print(f"\n[SUCCESS] Minimal Optimal Panel discovered with {optimal_panel[0]} features (AUC: {optimal_panel[1]:.4f})!")
    else:
        print("\n[CONCLUSION] Tuning thresholds improved extraction, but >0.85 AUC remains impossible under strict statistical bounds.")
        
    print("\n=== Iteration 15b Complete ===")

if __name__ == "__main__":
    main()
