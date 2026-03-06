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
    # Force parameters for Iteration 3
    # Non-linear SVM with explicit RFE feature selection
    Config.MODEL_TYPE = 'svm'
    Config.FEATURE_METHOD = 'rfe' 
    Config.setup()
    
    print(f"\n=== Iteration 3 Started: Non-linear SVM + RFE ===")
    
    loader = DataLoader()
    X, y, feature_names = loader.load_process()
    
    selector = FeatureSelector(X, y, feature_names, method=Config.FEATURE_METHOD)
    X_selected, selected_names = selector.execute()
    
    # SVM requires scaling for training as well
    from sklearn.preprocessing import StandardScaler
    X_selected_scaled = StandardScaler().fit_transform(X_selected)
    
    trainer = Trainer(X_selected_scaled, y, selected_names)
    final_model = trainer.run()
    
    iter_folder_name = f"iter_03_svm_rfe_{Config.TIMESTAMP}"
    backup_iteration(iter_folder_name)
    
    print(f"\n=== Iteration 3 Completed ===")

if __name__ == "__main__":
    main()
