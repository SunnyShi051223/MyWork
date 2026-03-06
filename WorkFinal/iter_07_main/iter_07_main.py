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
    
    # Backup Current Results (from experiments dir)
    res_dest = os.path.join(dest_path, 'results_backup')
    if os.path.exists(res_dest):
        shutil.rmtree(res_dest)
    shutil.copytree(Config.RESULT_DIR, res_dest)
    
    # Backup the wrapper script itself
    shutil.copy2(__file__, os.path.join(dest_path, os.path.basename(__file__)))
    
    print(f"[Backup] Done. Snapshot saved to: {dest_path}")

def main():
    # Force parameters for Iteration 7: The Utimate Ensemble
    Config.MODEL_TYPE = 'ensemble'
    Config.FEATURE_METHOD = 'consensus' 
    Config.setup()
    
    print(f"\n=== Iteration 7 Started: Consensus Features + Soft-Voting Ensemble ===")
    
    loader = DataLoader()
    X, y, feature_names = loader.load_process()
    
    selector = FeatureSelector(X, y, feature_names, method=Config.FEATURE_METHOD)
    X_selected, selected_names = selector.execute()
    
    # Run trainer (SMOTE-ENN is inside)
    trainer = Trainer(X_selected, y, selected_names)
    final_model = trainer.run()
    
    iter_folder_name = f"iter_07_ensemble_consensus_{Config.TIMESTAMP}"
    backup_iteration(iter_folder_name)
    
    print(f"\n=== Iteration 7 Completed ===")
    
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
