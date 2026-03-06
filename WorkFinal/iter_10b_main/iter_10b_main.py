import os
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, accuracy_score, confusion_matrix, average_precision_score
import csv

from src.config import Config
from src.data_loader import DataLoader
from src.model_factory import ModelFactory
from src.deep_learning import GANAugmenter

def backup_iteration(iter_name, metrics):
    print(f"\n[Backup] Saving Iteration {iter_name} snapshot...")
    dest_path = os.path.join(Config.BASE_DIR, 'results', iter_name)
    os.makedirs(dest_path, exist_ok=True)
    src_dest = os.path.join(dest_path, 'src_backup')
    if os.path.exists(src_dest): shutil.rmtree(src_dest)
    shutil.copytree(os.path.join(Config.BASE_DIR, 'src'), src_dest)
    res_dest = os.path.join(dest_path, 'results_backup')
    os.makedirs(res_dest, exist_ok=True)
    
    df = pd.DataFrame(metrics)
    df.loc['Mean'] = df.mean()
    df.to_csv(os.path.join(res_dest, 'cv_metrics_detail.csv'))
    
    shutil.copy2(__file__, os.path.join(dest_path, os.path.basename(__file__)))
    print(f"[Backup] Done. Snapshot saved to: {dest_path}")

def main():
    Config.MODEL_TYPE = 'xgboost'
    Config.ENABLE_FEATURE_ENGINEERING = False
    Config.setup()
    
    print(f"\n=== Iteration 10b Started: Tabular WGAN Minority Synthesis (STRICT CV) ===")
    
    loader = DataLoader()
    X, y, feature_names = loader.load_process()
    
    skf = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)
    print(f"[Step 3] Starting {Config.N_FOLDS}-Fold Cross Validation (Method: StratifiedKFold)...")
    
    metrics = {
        'auc': [], 'recall': [], 'precision': [], 'f1': [],
        'accuracy': [], 'specificity': [], 'aupr': [], 'threshold': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # --- Strict inner-loop GAN augmentation ---
        print(f"       Fold {fold + 1}: Training WGAN entirely on {len(X_train)} training samples to synthesize minority class...")
        min_mask = y_train == 1
        maj_mask = y_train == 0
        X_min, y_min = X_train[min_mask], y_train[min_mask]
        X_maj, y_maj = X_train[maj_mask], y_train[maj_mask]
        
        target_samples = len(X_maj)
        augmenter = GANAugmenter(input_dim=X.shape[1], target_samples=target_samples, epochs=100, seed=Config.SEED+fold)
        X_train_res, y_train_res = augmenter.fit_resample(X_min, X_maj, y_min, y_maj)
        
        # Model Training on strictly augmented isolated fold
        model = ModelFactory.get_model()
        model.fit(X_train_res, y_train_res)
        
        y_proba = model.predict_proba(X_val)[:, 1]
        
        best_thresh = 0.5
        best_f1 = 0
        for thresh in np.linspace(0.1, 0.9, 100):
            y_pred_tmp = (y_proba >= thresh).astype(int)
            f1_tmp = f1_score(y_val, y_pred_tmp)
            if f1_tmp > best_f1:
                best_f1 = f1_tmp
                best_thresh = thresh
                
        y_final_pred = (y_proba >= best_thresh).astype(int)
        
        auc = roc_auc_score(y_val, y_proba)
        aupr = average_precision_score(y_val, y_proba)
        acc = accuracy_score(y_val, y_final_pred)
        tn, fp, fn, tp = confusion_matrix(y_val, y_final_pred).ravel()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        metrics['auc'].append(auc)
        metrics['aupr'].append(aupr)
        metrics['recall'].append(recall)
        metrics['specificity'].append(specificity)
        metrics['precision'].append(precision)
        metrics['f1'].append(best_f1)
        metrics['accuracy'].append(acc)
        metrics['threshold'].append(best_thresh)
        
        print(f"       Fold {fold + 1} Result: AUC={auc:.3f} | Recall={recall:.3f} | Spec={specificity:.3f} | AUPR={aupr:.3f} (Th={best_thresh:.2f})")

    mean_metrics = {k: np.mean(v) for k, v in metrics.items()}
    print("\n       [Final Average Metrics]")
    print(f"       AUC:         {mean_metrics['auc']:.4f}")
    print(f"       Recall:      {mean_metrics['recall']:.4f}")
    print(f"       Specificity: {mean_metrics['specificity']:.4f}")
    print(f"       AUPR:        {mean_metrics['aupr']:.4f}")
    print(f"       F1-Score:    {mean_metrics['f1']:.4f}")
    print(f"       Accuracy:    {mean_metrics['accuracy']:.4f}")
    
    file_exists = os.path.isfile(Config.GLOBAL_LOG_PATH)
    with open(Config.GLOBAL_LOG_PATH, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['Timestamp', 'Model', 'AUC', 'Recall', 'Specificity', 'AUPR', 'Precision', 'F1', 'Acc', 'Threshold']
        if not file_exists:
            writer.writerow(header)
        writer.writerow([
            Config.TIMESTAMP, "wgan_xgboost_strict_cv",
            f"{mean_metrics['auc']:.4f}", f"{mean_metrics['recall']:.4f}", f"{mean_metrics['specificity']:.4f}",
            f"{mean_metrics['aupr']:.4f}", f"{mean_metrics['precision']:.4f}", f"{mean_metrics['f1']:.4f}",
            f"{mean_metrics['accuracy']:.4f}", f"{np.mean(metrics['threshold']):.2f}"
        ])

    iter_folder_name = f"iter_10b_tabular_wgan_strict_cv_{Config.TIMESTAMP}"
    backup_iteration(iter_folder_name, metrics)
    print(f"\n=== Iteration 10b Completed ===")
    
if __name__ == "__main__":
    main()
