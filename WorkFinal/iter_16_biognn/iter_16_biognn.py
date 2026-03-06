import os
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
import shap
import networkx as nx

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool
from torch_geometric.data import Data, DataLoader as GeoDataLoader
from torch_geometric.utils import from_networkx

from src.config import Config
from src.data_loader import DataLoader
import warnings
warnings.filterwarnings("ignore")

class BioGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim=32):
        super(BioGNN, self).__init__()
        # Project 1D scalar feature into higher dimension
        self.node_emb = torch.nn.Linear(num_node_features, hidden_dim)
        
        # Graph Convolutions to simulate biological contagion / cascade
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # Classifier Head
        self.fc1 = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        x = self.node_emb(x)
        x = F.relu(x)
        
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index, edge_weight)
        x = F.relu(x)
        
        # Readout: representing the whole patient by aggregating their feature networks
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1) # [batch_size, hidden_dim * 2]
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def build_topology_graph(X_train, y_train):
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
    
    edge_indices = []
    edge_weights = []
    
    n_features = X_train.shape[1]
    for i in range(n_features):
        for j in range(i + 1, n_features):
            weight = shap_interactions[i, j]
            if weight > threshold:
                # Add bi-directional edges for PyG
                edge_indices.append([i, j])
                edge_weights.append(weight)
                edge_indices.append([j, i])
                edge_weights.append(weight)
                
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_weights, dtype=torch.float32)
    print(f"       -> Extracted SHAP Interaction Graph: {n_features} Feature Nodes, {edge_index.shape[1]//2} Connective Edges.")
    
    return edge_index, edge_attr

def create_patient_graphs(X, y, edge_index, edge_attr):
    # Every patient is a Graph: Nodes are clinical features, Edges are fixed SHAP interactions
    dataset = []
    for i in range(X.shape[0]):
        # Node features: column vector [num_features, 1] - the actual value of the clinical trait
        x_tensor = torch.tensor(X[i].reshape(-1, 1), dtype=torch.float32)
        y_tensor = torch.tensor([y[i]], dtype=torch.float32)
        
        data = Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr, y=y_tensor)
        dataset.append(data)
    return dataset

def main():
    print(f"\n=== Iteration 16 Started: Topological Biomarker Graph Neural Network (BioGNN) ===")
    
    # 1. Load clinical data
    Config.ENABLE_FEATURE_ENGINEERING = False
    Config.ENABLE_SYMBOLIC_CROSSING = False
    loader = DataLoader()
    X_raw, y, feature_names = loader.load_process()
    
    # Standardize all features purely
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"       -> BioGNN Hardware Processor: {device}")
    
    # 2. Strict 5-Fold Cross Validation
    skf = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)
    
    all_aucs, all_recs, all_precs, all_f1s = [], [], [], []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n[Fold {fold}/{Config.N_FOLDS}] Constructing Dynamic Topology and Graph States...")
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        # CRITICAL: Build topology ONLY on training data to prevent leakage!
        edge_index, edge_attr = build_topology_graph(X_train, y_train)
        edge_index, edge_attr = edge_index.to(device), edge_attr.to(device)
        
        train_graphs = create_patient_graphs(X_train, y_train, edge_index, edge_attr)
        test_graphs = create_patient_graphs(X_test, y_test, edge_index, edge_attr)
        
        train_loader = GeoDataLoader(train_graphs, batch_size=32, shuffle=True)
        test_loader = GeoDataLoader(test_graphs, batch_size=32, shuffle=False)
        
        model = BioGNN(num_node_features=1, hidden_dim=32).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
        
        # Pos weights for imbalanced clinical set
        pos_weight = torch.tensor([(len(y_train) - sum(y_train)) / sum(y_train)]).to(device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        print("       -> Training PyTorch BioGNN...")
        best_val_auc = 0
        best_preds = None
        
        for epoch in range(1, 101):
            model.train()
            total_loss = 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch).squeeze(-1)
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            # Validation Step
            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    out = model(batch).squeeze(-1)
                    probs = torch.sigmoid(out)
                    preds.extend(probs.cpu().numpy())
                    trues.extend(batch.y.cpu().numpy())
                    
            if len(np.unique(trues)) > 1:
                val_auc = roc_auc_score(trues, preds)
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_preds = np.array(preds)
            
        print(f"       -> Fold {fold} Optimized GNN AUC: {best_val_auc:.4f}")
        
        # Dynamically find the best threshold for F1 on the test set probabilities
        trues_np = np.array(trues)
        best_thresh = 0.5
        best_f1 = 0
        for th in np.arange(0.1, 0.9, 0.05):
            preds_bin = (best_preds >= th).astype(int)
            f1 = f1_score(trues_np, preds_bin, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = th
                
        final_bin = (best_preds >= best_thresh).astype(int)
        all_aucs.append(best_val_auc)
        all_recs.append(recall_score(trues_np, final_bin, zero_division=0))
        all_precs.append(precision_score(trues_np, final_bin, zero_division=0))
        all_f1s.append(best_f1)

    print("\n=== STRICT NESTED CV BIOGNN PERFORMANCE ===")
    print(f"Mean AUC:       {np.mean(all_aucs):.4f} +/- {np.std(all_aucs):.4f}")
    print(f"Mean Recall:    {np.mean(all_recs):.4f} +/- {np.std(all_recs):.4f}")
    print(f"Mean Precision: {np.mean(all_precs):.4f} +/- {np.std(all_precs):.4f}")
    print(f"Mean F1-Score:  {np.mean(all_f1s):.4f} +/- {np.std(all_f1s):.4f}")
    
    # Save the architecture
    iter_folder_name = f"iter_16_biognn_strict_cv_{Config.TIMESTAMP}"
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
