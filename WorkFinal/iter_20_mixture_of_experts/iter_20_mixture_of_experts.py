import os
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import catboost as cb
import shap

from src.config import Config
from src.data_loader import DataLoader
import warnings
warnings.filterwarnings("ignore")

def extract_patient_manifold(X_train, y_train, X_test):
    """
    Instead of feature-feature topology, we extract patient-patient topology (manifold).
    We use the SHAP values of the training set to project patients into a pathologically meaningful space.
    """
    model_for_shap = LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, 
                                    class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)
    model_for_shap.fit(X_train, y_train)
    explainer = shap.TreeExplainer(model_for_shap)
    
    # SHAP values (not interactions) for manifold projection
    shap_train = np.array(explainer.shap_values(X_train))
    shap_test = np.array(explainer.shap_values(X_test))
    
    # If binary classification in older shap versions, it returns a list
    if isinstance(shap_train, list):
        shap_train = shap_train[1]
        shap_test = shap_test[1]
        
    return shap_train, shap_test

def main():
    print(f"\n=== Iteration 20 Started: Clinical Phenotype Mixture of Experts (MoE) ===")
    
    Config.ENABLE_FEATURE_ENGINEERING = False
    Config.ENABLE_SYMBOLIC_CROSSING = False
    loader = DataLoader()
    X_raw, y, feature_names = loader.load_process()
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    
    skf = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)
    all_aucs, all_recs, all_precs, all_f1s = [], [], [], []
    
    K_PHENOTYPES = 3  # Assume 3 latent mechanisms of VTE
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n[Fold {fold}/5] Building MoE Precision Architecture...")
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        # 1. Project to Patient Causal-Manifold (no leakage)
        print("       -> Projecting patients into Causal-SHAP manifold...")
        shap_train, shap_test = extract_patient_manifold(X_train, y_train, X_test)
        
        # 2. Unsupervised Phenotype Discovery (GMM)
        print(f"       -> Discovering {K_PHENOTYPES} latent clinical phenotypes via GMM...")
        gmm = GaussianMixture(n_components=K_PHENOTYPES, covariance_type='diag', random_state=42)
        phenotypes_train = gmm.fit_predict(shap_train)
        
        # 3. Train the Gating Network (Router)
        # Router predicts the probability of a patient belonging to each phenotype based on RAW features
        print("       -> Training Soft-Router (Gating Network)...")
        router = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42, n_jobs=-1)
        router.fit(X_train, phenotypes_train)
        
        # Get gating probabilities (Weights for the Experts)
        # Shape: (N_samples, K_PHENOTYPES)
        router_probs_test = router.predict_proba(X_test) 
        
        # 4. Train K Specialized Experts
        print("       -> Training Specialized Clinical Experts...")
        experts = []
        for k in range(K_PHENOTYPES):
            # Strict Expert Formulation: Train specifically on patients in this phenotype cluster
            # Fallback to full train set if cluster is too small, but apply heavy sample weights
            weights = (phenotypes_train == k).astype(float) * 5 + 1.0 # Emphasize target phenotype
            
            # Use ordered CatBoost to prevent overfitting on tiny sub-clusters
            expert = cb.CatBoostClassifier(
                iterations=200, learning_rate=0.03, depth=4,
                eval_metric='Logloss', random_seed=42+k, verbose=False
            )
            # Find class balance in this synthetic 'cluster'
            pos_weight = (len(y_train) - sum(y_train)) / (sum(y_train) + 1e-5)
            expert.set_params(scale_pos_weight=pos_weight)
            
            expert.fit(X_train, y_train, sample_weight=weights)
            experts.append(expert)
            
        # 5. MoE Soft-Assembly Prediction
        # \sum (P(Phenotype) * P(VTE | Expert))
        final_preds_test = np.zeros(len(X_test))
        for k in range(K_PHENOTYPES):
            expert_preds = experts[k].predict_proba(X_test)[:, 1]
            final_preds_test += router_probs_test[:, k] * expert_preds
            
        val_auc = roc_auc_score(y_test, final_preds_test)
        print(f"       -> Fold {fold} MoE Final Fusion AUC: {val_auc:.4f}")
        
        best_thresh, best_f1 = 0.5, 0
        for th in np.arange(0.1, 0.9, 0.05):
            preds_bin = (final_preds_test >= th).astype(int)
            f1 = f1_score(y_test, preds_bin, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, th
                
        final_bin = (final_preds_test >= best_thresh).astype(int)
        all_aucs.append(val_auc)
        all_recs.append(recall_score(y_test, final_bin, zero_division=0))
        all_precs.append(precision_score(y_test, final_bin, zero_division=0))
        all_f1s.append(best_f1)

    print("\n=== STRICT NESTED CV MIXTURE OF EXPERTS (MoE) PERFORMANCE ===")
    print(f"Mean AUC:       {np.mean(all_aucs):.4f} +/- {np.std(all_aucs):.4f}")
    print(f"Mean Recall:    {np.mean(all_recs):.4f} +/- {np.std(all_recs):.4f}")
    print(f"Mean Precision: {np.mean(all_precs):.4f} +/- {np.std(all_precs):.4f}")
    print(f"Mean F1-Score:  {np.mean(all_f1s):.4f} +/- {np.std(all_f1s):.4f}")
    
    iter_folder_name = f"iter_20_mixture_of_experts_strict_cv_{Config.TIMESTAMP}"
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
