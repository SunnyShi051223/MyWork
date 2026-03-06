import os
import shutil
from src.config import Config
from src.data_loader import DataLoader
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
    Config.MODEL_TYPE = 'stacking' 
    Config.FEATURE_METHOD = 'genetic_algorithm' 
    Config.ENABLE_FEATURE_ENGINEERING = False 
    Config.ENABLE_SYMBOLIC_CROSSING = True 
    Config.ENABLE_WGAN_AUGMENTATION = True 
    Config.setup()
    
    print(f"\n=== Iteration 14 Started: Evolutionary Non-Linear Manifold (Symbolic Crossing + Genetic Algorithm) ===")
    print(f"!!! STRICT NESTED CV ENABLED: Exploring vast high-order polynomial spaces without leakage !!!")
    
    loader = DataLoader()
    X, y, feature_names = loader.load_process()
    
    trainer = Trainer(X, y, feature_names, enable_nested_fs=True)
    final_model = trainer.run()
    
    iter_folder_name = f"iter_14_genetic_symbolic_strict_cv_{Config.TIMESTAMP}"
    backup_iteration(iter_folder_name)
    print(f"\n=== Iteration 14 Completed ===")

if __name__ == "__main__":
    main()
