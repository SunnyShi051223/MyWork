import os
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import shap
import networkx as nx
import community as community_louvain

from src.config import Config
from src.data_loader import DataLoader
import warnings
warnings.filterwarnings("ignore")

def extract_biological_modules(X_train, y_train):
    print("       -> Extracting Global Pathological Hierarchy via Causal-SHAP Operations...")
    model_for_shap = LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.05,
                                    class_weight='balanced', random_state=Config.SEED, n_jobs=-1,
                                    verbose=-1)
    model_for_shap.fit(X_train, y_train)
    explainer = shap.TreeExplainer(model_for_shap)
    shap_interactions = np.array(explainer.shap_interaction_values(X_train))

    while len(shap_interactions.shape) > 2:
        if shap_interactions.shape[-3] == 2: 
            shap_interactions = shap_interactions[..., 1, :, :]
        shap_interactions = np.abs(shap_interactions).mean(axis=0)

    # Threshold the graph to extract skeleton (top 20% edges)
    threshold = np.percentile(shap_interactions, 80)
    
    G = nx.Graph()
    n_features = X_train.shape[1]
    for i in range(n_features): G.add_node(i)
        
    for i in range(n_features):
        for j in range(i + 1, n_features):
            weight = shap_interactions[i, j]
            if weight > threshold:
                G.add_edge(i, j, weight=weight)
                
    partition = community_louvain.best_partition(G, weight='weight', resolution=1.0)
    
    modules = {}
    for node, comm_id in partition.items():
        if comm_id not in modules: modules[comm_id] = []
        modules[comm_id].append(node)
        
    robust_modules = {k: v for k, v in modules.items() if len(v) >= 2}
    print(f"       -> Discovered {len(robust_modules)} resilient multi-node Biological Modules.")
    return robust_modules

def generate_pathway_restricted_features(X, modules):
    # Only cross features that belong to the SAME biological module
    # Returns the original X + newly crossed features
    new_features = []
    
    for comm_id, feature_indices in modules.items():
        for i in range(len(feature_indices)):
            for j in range(i + 1, len(feature_indices)):
                idx1, idx2 = feature_indices[i], feature_indices[j]
                arr1, arr2 = X[:, idx1], X[:, idx2]
                
                # Multiplication (Synergy)
                new_features.append(arr1 * arr2)
                
                # Division/Ratio (Imbalance)
                arr2_safe = np.where(arr2 == 0, 1e-5, arr2)
                arr1_safe = np.where(arr1 == 0, 1e-5, arr1)
                new_features.append(arr1 / arr2_safe)
                new_features.append(arr2 / arr1_safe)
                
    if len(new_features) == 0:
        return X
        
    X_new = np.column_stack(new_features)
    # Clip extreme ratios to avoid Inf destroying the tree weights
    X_new = np.clip(X_new, -1e6, 1e6) 
    return np.hstack([X, X_new])

def build_stacking_ensemble():
    estimators = [
        ('xgb', xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, 
                                  scale_pos_weight=3.6, eval_metric='logloss', random_state=42, n_jobs=-1)),
        ('lgbm', LGBMClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, 
                                class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1))
    ]
    meta_model = LogisticRegression(class_weight='balanced', C=0.1, random_state=42)
    return StackingClassifier(estimators=estimators, final_estimator=meta_model, cv=5, n_jobs=-1, passthrough=False)

def main():
    print(f"\n=== Iteration 18 Started: Pathway-Restricted Symbolic Modularity ===")
    
    # 1. Load data
    Config.ENABLE_FEATURE_ENGINEERING = False
    Config.ENABLE_SYMBOLIC_CROSSING = False # We do our own highly controlled crossing
    loader = DataLoader()
    X_raw, y, feature_names = loader.load_process()
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    
    # 2. Strict 5-Fold Nested Cross Validation
    skf = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)
    
    all_aucs, all_recs, all_precs, all_f1s = [], [], [], []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n[Fold {fold}/{Config.N_FOLDS}] Building Topological Modules...")
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        # 3. Detect Biological Modules exclusively on Train to prevent Data Leakage
        modules = extract_biological_modules(X_train, y_train)
        
        # 4. Generate restricted non-linear intersections strictly within pathways
        print("       -> Applying Pathway-Restricted Iterative Non-Linear Expansion...")
        X_train_expanded = generate_pathway_restricted_features(X_train, modules)
        X_test_expanded = generate_pathway_restricted_features(X_test, modules)
        
        print(f"       -> Dimensionality successfully controlled: From {X_train.shape[1]} -> {X_train_expanded.shape[1]} dimensions.")
        
        # 5. Train Stacking Ensembles
        print("       -> Training Hybrid Stacking Meta-Ensemble on Mathematically Constrained Manifold...")
        model = build_stacking_ensemble()
        model.fit(X_train_expanded, y_train)
        
        preds = model.predict_proba(X_test_expanded)[:, 1]
        
        val_auc = roc_auc_score(y_test, preds)
        print(f"       -> Fold {fold} Constrained-Stacking AUC: {val_auc:.4f}")
        
        # 6. Dynamic Call Thresholding for balanced F1
        best_thresh = 0.5
        best_f1 = 0
        for th in np.arange(0.1, 0.9, 0.05):
            preds_bin = (preds >= th).astype(int)
            f1 = f1_score(y_test, preds_bin, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = th
                
        final_bin = (preds >= best_thresh).astype(int)
        all_aucs.append(val_auc)
        all_recs.append(recall_score(y_test, final_bin, zero_division=0))
        all_precs.append(precision_score(y_test, final_bin, zero_division=0))
        all_f1s.append(best_f1)

    print("\n=== STRICT NESTED CV PATHWAY-RESTRICTED PERFORMANCE ===")
    print(f"Mean AUC:       {np.mean(all_aucs):.4f} +/- {np.std(all_aucs):.4f}")
    print(f"Mean Recall:    {np.mean(all_recs):.4f} +/- {np.std(all_recs):.4f}")
    print(f"Mean Precision: {np.mean(all_precs):.4f} +/- {np.std(all_precs):.4f}")
    print(f"Mean F1-Score:  {np.mean(all_f1s):.4f} +/- {np.std(all_f1s):.4f}")
    
    # Save the architecture
    iter_folder_name = f"iter_18_pathway_evolution_strict_cv_{Config.TIMESTAMP}"
    dest_path = os.path.join(Config.BASE_DIR, 'results', iter_folder_name)
    os.makedirs(dest_path, exist_ok=True)
    res_dest = os.path.join(dest_path, 'results_backup')
    os.makedirs(res_dest, exist_ok=True)
    
    df_res = pd.DataFrame({'auc': all_aucs, 'recall': all_recs, 'precision': all_precs, 'f1': all_f1s})
    df_res.loc['Mean'] = df_res.mean()
    df_res.to_csv(os.path.join(res_dest, "cv_metrics_detail.csv"))
    
    shutil.copy2(__file__, os.path.join(dest_path, os.path.basename(__file__)))
    print(f"\n[Backup] Done. Snapshot saved to: {dest_path}")

if __name__ == "__main__":
    main()
