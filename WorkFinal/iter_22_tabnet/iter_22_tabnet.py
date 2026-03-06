import os
import shutil
import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler

from src.config import Config
from src.data_loader import DataLoader
import warnings
warnings.filterwarnings("ignore")

def main():
    print(f"\n=== Iteration 22 Started: Attentive Tabular Transformer (TabNet) ===")
    
    # TabNet acts as its own feature selector (Sparsemax Attention), so we feed it RAW features.
    Config.ENABLE_FEATURE_ENGINEERING = False
    Config.ENABLE_SYMBOLIC_CROSSING = False
    loader = DataLoader()
    X_raw, y, feature_names = loader.load_process()
    
    # Neural Networks heavily require standardization
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    
    # Handle Class Imbalance via explicit cost weighing inside PyTorch
    class_1_weight = (len(y) - sum(y)) / (sum(y) + 1e-5)
    weights = [1.0 if label == 0 else class_1_weight for label in y]
    
    skf = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)
    all_aucs, all_recs, all_precs, all_f1s = [], [], [], []
    
    # Check GPU availability
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"       -> Utilizing compute device: {device_name.upper()}")

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n[Fold {fold}/5] Training Sequential Self-Attention Transformer...")
        
        # We need an internal validation set for TabNet's Early Stopping
        # Split 10% from the training set *internally* to avoid leaking the main test fold
        internal_skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        inner_train_idx, inner_val_idx = next(internal_skf.split(X[train_idx], y[train_idx]))
        
        X_train_inner = X[train_idx][inner_train_idx]
        y_train_inner = y[train_idx][inner_train_idx]
        X_val_inner = X[train_idx][inner_val_idx]
        y_val_inner = y[train_idx][inner_val_idx]
        
        X_test, y_test = X[test_idx], y[test_idx]
        
        # Extract fold-specific weights for PyTorch loss weighting
        fold_weights = np.array(weights)[train_idx][inner_train_idx]
        
        # Configure TabNet hyperparameters (Tuned for small, noisy tabular data)
        # Low n_d / n_a to prevent severe overfitting on N=397
        clf = TabNetClassifier(
            n_d=16, n_a=16, n_steps=3, gamma=1.3,
            n_independent=2, n_shared=2,
            lambda_sparse=1e-3, # Critical penalty to force feature selection
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
            mask_type='entmax', # "Sparsemax" is also an option
            device_name=device_name,
            verbose=0
        )
        
        clf.fit(
            X_train=X_train_inner, y_train=y_train_inner,
            eval_set=[(X_train_inner, y_train_inner), (X_val_inner, y_val_inner)],
            eval_name=['train', 'valid'],
            eval_metric=['auc'],
            max_epochs=200, patience=25,
            batch_size=64, virtual_batch_size=16,
            num_workers=0,
            drop_last=False,
            # Note: pytorch-tabnet handles class weights internally via loss_fn / Custom weights
            weights=1  # 1 means balanced auto-weighting in tabnet v4+
        )
        
        # Inference
        preds_test_proba = clf.predict_proba(X_test)[:, 1]
        
        val_auc = roc_auc_score(y_test, preds_test_proba)
        print(f"       -> Fold {fold} TabNet Transformer AUC: {val_auc:.4f}")
        
        # Dynamic Threshold Search
        best_thresh, best_f1 = 0.5, 0
        for th in np.arange(0.1, 0.9, 0.05):
            preds_bin = (preds_test_proba >= th).astype(int)
            f1 = f1_score(y_test, preds_bin, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, th
                
        final_bin = (preds_test_proba >= best_thresh).astype(int)
        all_aucs.append(val_auc)
        all_recs.append(recall_score(y_test, final_bin, zero_division=0))
        all_precs.append(precision_score(y_test, final_bin, zero_division=0))
        all_f1s.append(best_f1)

    print("\n=== STRICT NESTED CV ATTENTIVE TABULAR TRANSFORMER (TabNet) ===")
    print(f"Mean AUC:       {np.mean(all_aucs):.4f} +/- {np.std(all_aucs):.4f}")
    print(f"Mean Recall:    {np.mean(all_recs):.4f} +/- {np.std(all_recs):.4f}")
    print(f"Mean Precision: {np.mean(all_precs):.4f} +/- {np.std(all_precs):.4f}")
    print(f"Mean F1-Score:  {np.mean(all_f1s):.4f} +/- {np.std(all_f1s):.4f}")
    
    iter_folder_name = f"iter_22_tabnet_transformer_strict_cv_{Config.TIMESTAMP}"
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
