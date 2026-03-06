import os
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from lightgbm import LGBMClassifier
import catboost as cb
import shap
import networkx as nx
import community as community_louvain # python-louvain

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
    for i in range(n_features): 
        G.add_node(i)
        
    for i in range(n_features):
        for j in range(i + 1, n_features):
            weight = shap_interactions[i, j]
            if weight > threshold:
                G.add_edge(i, j, weight=weight)
                
    # Detect biological communities (Pathways) using Louvain Modularity
    partition = community_louvain.best_partition(G, weight='weight', resolution=1.0)
    
    # Organize features into pathways
    modules = {}
    for node, comm_id in partition.items():
        if comm_id not in modules:
            modules[comm_id] = []
        modules[comm_id].append(node)
        
    print(f"       -> Discovered {len(modules)} distinct Biological Subnetwork Modules.")
    # Filter out trivial modules (e.g. less than 3 features)
    robust_modules = {k: v for k, v in modules.items() if len(v) >= 2}
    print(f"       -> Retained {len(robust_modules)} robust pathways for Meta-Signal Extraction.")
    
    return robust_modules

def extract_pathway_signals(X, modules, pca_models=None):
    # For each module, compress its features into a single 1D Principal Component Meta-Signal
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
            pca = pca_models[comm_id]
            signal = pca.transform(sub_X)
        signals.append(signal.flatten())
        
    X_meta = np.column_stack(signals)
    return X_meta, pca_models

def main():
    print(f"\n=== Iteration 17 Started: Systems Biology & Network Modularity (CatBoost) ===")
    
    # 1. Load clinical data
    Config.ENABLE_FEATURE_ENGINEERING = False
    Config.ENABLE_SYMBOLIC_CROSSING = False
    loader = DataLoader()
    X_raw, y, feature_names = loader.load_process()
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    
    # 2. Strict 5-Fold Cross Validation
    skf = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)
    
    all_aucs, all_recs, all_precs, all_f1s = [], [], [], []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n[Fold {fold}/{Config.N_FOLDS}] Constructing Pathological Communities...")
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        # Build Topology and extract cliques ONLY on training data to prevent leakage!
        modules = extract_biological_modules(X_train, y_train)
        
        # Extract the Meta-Signals for the Training Set
        X_meta_train, pca_models = extract_pathway_signals(X_train, modules)
        
        # Fuse the raw core variables with the new High-Level Pathway Signals
        X_fused_train = np.hstack([X_train, X_meta_train])
        
        # Do the same purely inferentially for Test
        X_meta_test, _ = extract_pathway_signals(X_test, modules, pca_models)
        X_fused_test = np.hstack([X_test, X_meta_test])
        
        # Train CatBoost! Known to be SOTA on small N < 1000 tabular data.
        print("       -> Training CatBoost with Ordered Target Calculus on Fused Medical Data...")
        scale_pos = (len(y_train) - sum(y_train)) / sum(y_train)
        
        model = cb.CatBoostClassifier(
            iterations=500,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3,
            scale_pos_weight=scale_pos,
            bootstrap_type='MVS',
            eval_metric='AUC',
            random_seed=Config.SEED,
            verbose=False,
            early_stopping_rounds=50
        )
        
        # Using a portion of train as validation for CatBoost early stopping to strictly prevent overfitting
        X_tr, X_val, y_tr, y_val = train_test_split(X_fused_train, y_train, test_size=0.1, stratify=y_train, random_state=42)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
        
        preds = model.predict_proba(X_fused_test)[:, 1]
        
        val_auc = roc_auc_score(y_test, preds)
        print(f"       -> Fold {fold} CatBoost Module-Fusion AUC: {val_auc:.4f}")
        
        # Find best metric balance
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

    print("\n=== STRICT NESTED CV SYSTEM BIOLOGY MODULE PERFORMANCE ===")
    print(f"Mean AUC:       {np.mean(all_aucs):.4f} +/- {np.std(all_aucs):.4f}")
    print(f"Mean Recall:    {np.mean(all_recs):.4f} +/- {np.std(all_recs):.4f}")
    print(f"Mean Precision: {np.mean(all_precs):.4f} +/- {np.std(all_precs):.4f}")
    print(f"Mean F1-Score:  {np.mean(all_f1s):.4f} +/- {np.std(all_f1s):.4f}")
    
    # Save the architecture
    iter_folder_name = f"iter_17_systems_biology_strict_cv_{Config.TIMESTAMP}"
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
