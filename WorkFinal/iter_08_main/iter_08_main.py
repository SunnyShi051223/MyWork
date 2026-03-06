import os
import shutil
from datetime import datetime
from src.config import Config
from src.data_loader import DataLoader
from src.feature_selection import FeatureSelector
from src.trainer import Trainer

def backup_iteration(iter_name):
    print(f"\n[Backup] Saving Iteration {iter_name} snapshot...")
    dest_path = os.path.join(Config.BASE_DIR, 'results', iter_name)
    os.makedirs(dest_path, exist_ok=True)
    
    # Backup Source Code
    src_dest = os.path.join(dest_path, 'src_backup')
    if os.path.exists(src_dest):
        shutil.rmtree(src_dest)
    shutil.copytree(os.path.join(Config.BASE_DIR, 'src'), src_dest)
    
    # Backup Current Results
    res_dest = os.path.join(dest_path, 'results_backup')
    if os.path.exists(res_dest):
        shutil.rmtree(res_dest)
    shutil.copytree(Config.RESULT_DIR, res_dest)
    
    # Backup the wrapper script itself
    shutil.copy2(__file__, os.path.join(dest_path, os.path.basename(__file__)))
    
    print(f"[Backup] Done. Snapshot saved to: {dest_path}")

def main():
    # Force parameters for Iteration 8: Clinical Feature Engineering
    Config.MODEL_TYPE = 'xgboost'
    Config.FEATURE_METHOD = 'rfecv' # Let RFECV select the best among raw + new composite features
    Config.ENABLE_FEATURE_ENGINEERING = True # <<< The Core Paradigm Shift
    Config.setup()
    
    print(f"\n=== Iteration 8 Started: Clinical Subtyping & Risk Composites (STRICT CV) ===")
    
    loader = DataLoader()
    X, y, feature_names = loader.load_process()
    
    # Feature Selection occurs strictly inside the Trainer's CV fold
    trainer = Trainer(X, y, feature_names, enable_nested_fs=True)
    final_model = trainer.run()
    
    iter_folder_name = f"iter_08_clinical_subtype_{Config.TIMESTAMP}"
    backup_iteration(iter_folder_name)
    
    print(f"\n=== Iteration 8 Completed ===")
    
    # Explicitly calculate and print the average scores 
    import pandas as pd
    res_path = os.path.join(Config.RESULT_DIR, 'cv_metrics_detail.csv')
    df = pd.read_csv(res_path, index_col=0)
    print("\n[Summary for Direct Output]")
    print(f"Mean AUC:       {df.loc['Mean', 'auc']:.4f}")
    print(f"Mean Recall:    {df.loc['Mean', 'recall']:.4f}")
    print(f"Mean Precision: {df.loc['Mean', 'precision']:.4f}")

if __name__ == "__main__":
    main()
