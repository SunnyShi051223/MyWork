import os
import shutil
from src.config import Config
from src.data_loader import DataLoader
from src.feature_selection import FeatureSelector
from src.trainer import Trainer

def backup_iteration(iter_name):
    print(f"\n[Backup] Saving Iteration {iter_name} snapshot...")
    dest_path = os.path.join(Config.BASE_DIR, 'results', iter_name)
    os.makedirs(dest_path, exist_ok=True)
    src_dest = os.path.join(dest_path, 'src_backup')
    if os.path.exists(src_dest): shutil.rmtree(src_dest)
    shutil.copytree(os.path.join(Config.BASE_DIR, 'src'), src_dest)
    res_dest = os.path.join(dest_path, 'results_backup')
    if os.path.exists(res_dest): shutil.rmtree(res_dest)
    shutil.copytree(Config.RESULT_DIR, res_dest)
    shutil.copy2(__file__, os.path.join(dest_path, os.path.basename(__file__)))
    print(f"[Backup] Done. Snapshot saved to: {dest_path}")

def main():
    Config.MODEL_TYPE = 'xgboost'
    Config.FEATURE_METHOD = 'shap_network' 
    Config.ENABLE_FEATURE_ENGINEERING = False # Test pure algorithm
    Config.setup()
    
    print(f"\n=== Iteration 9a Started: SHAP Interaction Network Topology (STRICT CV) ===")
    
    loader = DataLoader()
    X, y, feature_names = loader.load_process()
    
    trainer = Trainer(X, y, feature_names, enable_nested_fs=True)
    final_model = trainer.run()
    
    iter_folder_name = f"iter_09a_shap_network_{Config.TIMESTAMP}"
    backup_iteration(iter_folder_name)
    
    print(f"\n=== Iteration 9a Completed ===")
    
if __name__ == "__main__":
    main()
