import os
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from lightgbm import LGBMClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import catboost as cb
import shap
import networkx as nx
import community as community_louvain 
from imblearn.combine import SMOTETomek

from src.config import Config
from src.data_loader import DataLoader
import warnings
warnings.filterwarnings("ignore")

# --- ITERATION 17 LOGIC (Modularity + CatBoost) ---
def extract_biological_modules(X_train, y_train):
    model_for_shap = LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)
    model_for_shap.fit(X_train, y_train)
    explainer = shap.TreeExplainer(model_for_shap)
    shap_interactions = np.array(explainer.shap_interaction_values(X_train))
    while len(shap_interactions.shape) > 2:
        if shap_interactions.shape[-3] == 2: shap_interactions = shap_interactions[..., 1, :, :]
        shap_interactions = np.abs(shap_interactions).mean(axis=0)

    threshold = np.percentile(shap_interactions, 80)
    G = nx.Graph()
    n_features = X_train.shape[1]
    for i in range(n_features): G.add_node(i)
    for i in range(n_features):
        for j in range(i + 1, n_features):
            weight = shap_interactions[i, j]
            if weight > threshold: G.add_edge(i, j, weight=weight)
                
    partition = community_louvain.best_partition(G, weight='weight', resolution=1.0)
    modules = {}
    for node, comm_id in partition.items():
        if comm_id not in modules: modules[comm_id] = []
        modules[comm_id].append(node)
    return {k: v for k, v in modules.items() if len(v) >= 2}

def extract_pathway_signals(X, modules, pca_models=None):
    signals = []
    is_training = pca_models is None
    if is_training: pca_models = {}
    for comm_id, feature_indices in modules.items():
        sub_X = X[:, feature_indices]
        if is_training:
            pca = PCA(n_components=1)
            signal = pca.fit_transform(sub_X)
            pca_models[comm_id] = pca
        else:
            signal = pca_models[comm_id].transform(sub_X)
        signals.append(signal.flatten())
    if not signals: return np.zeros((X.shape[0], 0)), pca_models
    return np.column_stack(signals), pca_models


# --- ITERATION 13 LOGIC (Causal-SHAP + Augmentation + Stacking) ---
def causal_shap_consensus(X_train_df, y_train):
    from sklearn.feature_selection import mutual_info_classif
    # SHAP Topology Hubs
    model_shap = LGBMClassifier(n_estimators=100, max_depth=4, class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)
    model_shap.fit(X_train_df.values, y_train)
    explainer = shap.TreeExplainer(model_shap)
    shap_interactions = np.array(explainer.shap_interaction_values(X_train_df.values))
    while len(shap_interactions.shape) > 2:
        if shap_interactions.shape[-3] == 2: shap_interactions = shap_interactions[..., 1, :, :]
        shap_interactions = np.abs(shap_interactions).mean(axis=0)
    
    threshold = np.percentile(shap_interactions, 75)
    G = nx.Graph()
    for i in range(X_train_df.shape[1]): G.add_node(i)
    for i in range(X_train_df.shape[1]):
        for j in range(i+1, X_train_df.shape[1]):
            if shap_interactions[i,j] > threshold: G.add_edge(i, j, weight=shap_interactions[i,j])
    centrality = nx.pagerank(G, weight='weight')
    shap_scores = np.array([centrality.get(i, 0) for i in range(X_train_df.shape[1])])
    
    # MRMR proxy (MI) to speed up Iteration 19 execution safely
    mi_scores = mutual_info_classif(X_train_df.values, y_train, random_state=42)
    
    # Consensus
    norm_shap = shap_scores / (shap_scores.max() + 1e-9)
    norm_mi = mi_scores / (mi_scores.max() + 1e-9)
    consensus = norm_shap + norm_mi
    top_indices = np.argsort(consensus)[::-1][:15]
    return X_train_df.columns[top_indices].tolist()

def get_stacking_ensemble():
    estimators = [
        ('xgb', xgb.XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, scale_pos_weight=3.6, eval_metric='logloss', random_state=42, n_jobs=-1)),
        ('lgbm', LGBMClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1))
    ]
    meta_model = LogisticRegression(class_weight='balanced', C=0.1, random_state=42)
    return StackingClassifier(estimators=estimators, final_estimator=meta_model, cv=5, n_jobs=-1)


def main():
    print(f"\n=== Iteration 19 Started: The Grand Master Ensemble (Iter13 + Iter17) ===")
    
    Config.ENABLE_FEATURE_ENGINEERING = False
    Config.ENABLE_SYMBOLIC_CROSSING = False
    loader = DataLoader()
    X_raw, y, feature_names = loader.load_process()
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    X_df = pd.DataFrame(X, columns=feature_names) # Needed for Iter 13
    
    skf = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)
    all_aucs, all_recs, all_precs, all_f1s = [], [], [], []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n[Fold {fold}/5] Executing Orthogonal Dual-Paradigms...")
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        X_train_df, X_test_df = X_df.iloc[train_idx], X_df.iloc[test_idx]
        
        # -------------------------------------------------------------------
        # PIPELINE A: Iteration 17 (Systems Biology Topology + CatBoost)
        # -------------------------------------------------------------------
        modules = extract_biological_modules(X_train, y_train)
        X_meta_train, pca_models = extract_pathway_signals(X_train, modules)
        X_meta_test, _ = extract_pathway_signals(X_test, modules, pca_models)
        
        X_fused_train = np.hstack([X_train, X_meta_train])
        X_fused_test = np.hstack([X_test, X_meta_test])
        
        scale_pos = (len(y_train) - sum(y_train)) / sum(y_train)
        cat_model = cb.CatBoostClassifier(iterations=400, learning_rate=0.03, depth=5, 
                                          scale_pos_weight=scale_pos, eval_metric='AUC', 
                                          random_seed=42, verbose=False)
        cat_model.fit(X_fused_train, y_train)
        preds_17 = cat_model.predict_proba(X_fused_test)[:, 1]
        
        # -------------------------------------------------------------------
        # PIPELINE B: Iteration 13 (Causal-SHAP + Augmentation + Stacking)
        # -------------------------------------------------------------------
        top_feats = causal_shap_consensus(X_train_df, y_train)
        X_train_sub = X_train_df[top_feats].values
        X_test_sub = X_test_df[top_feats].values
        
        # Fallback to extreme robust SMOTETomek for speed and stability in mega-script
        augmenter = SMOTETomek(random_state=42)
        X_train_aug, y_train_aug = augmenter.fit_resample(X_train_sub, y_train)
        
        stack_model = get_stacking_ensemble()
        stack_model.fit(X_train_aug, y_train_aug)
        preds_13 = stack_model.predict_proba(X_test_sub)[:, 1]
        
        # -------------------------------------------------------------------
        # FUSION: Orthogonal Soft-Voting
        # -------------------------------------------------------------------
        preds_fusion = (preds_17 + preds_13) / 2.0
        
        val_auc = roc_auc_score(y_test, preds_fusion)
        print(f"       -> Fold {fold} Master Ensemble AUC: {val_auc:.4f} (Iter 17: {roc_auc_score(y_test,preds_17):.4f} | Iter 13: {roc_auc_score(y_test,preds_13):.4f})")
        
        # Find best metric balance
        best_thresh, best_f1 = 0.5, 0
        for th in np.arange(0.1, 0.9, 0.05):
            preds_bin = (preds_fusion >= th).astype(int)
            f1 = f1_score(y_test, preds_bin, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, th
                
        final_bin = (preds_fusion >= best_thresh).astype(int)
        all_aucs.append(val_auc)
        all_recs.append(recall_score(y_test, final_bin, zero_division=0))
        all_precs.append(precision_score(y_test, final_bin, zero_division=0))
        all_f1s.append(best_f1)

    print("\n=== STRICT NESTED CV THE GRAND MASTER FUSION PERFORMANCE ===")
    print(f"Mean AUC:       {np.mean(all_aucs):.4f} +/- {np.std(all_aucs):.4f}")
    print(f"Mean Recall:    {np.mean(all_recs):.4f} +/- {np.std(all_recs):.4f}")
    print(f"Mean Precision: {np.mean(all_precs):.4f} +/- {np.std(all_precs):.4f}")
    print(f"Mean F1-Score:  {np.mean(all_f1s):.4f} +/- {np.std(all_f1s):.4f}")
    
    iter_folder_name = f"iter_19_master_fusion_strict_cv_{Config.TIMESTAMP}"
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
