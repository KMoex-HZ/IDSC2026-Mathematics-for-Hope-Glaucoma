import os
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, confusion_matrix
from src.data.dataset import GlaucomaDataset
from src.data.dataloader import train_transform, val_transform, get_train_test_splits
from src.models.model import GlaucomaEfficientNet, WeightedQualityBCE

def set_seed(seed=42):
    """Function to lock all randomness to ensure 100% reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed locked at: {seed}")

def train_ultimate_pipeline():
    # Set seed at the beginning to lock data splits and weight initialization
    set_seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = 'data/raw/hillel-yaffe-glaucoma-dataset-hygd-a-gold-standard-annotated-fundus-dataset-for-glaucoma-detection-1.0.0'
    csv_path = os.path.join(base_dir, 'Labels.csv')
    img_dir = os.path.join(base_dir, 'Images')
    
    # 1. Separate 20% of Patients for a fixed Test Set
    df_full, df_train_val, df_test, cv_splits, train_val_idx, test_idx = get_train_test_splits(csv_path)
    
    # Loader for the Final Evaluation (Test Set)
    test_dataset = GlaucomaDataset(csv_path, img_dir, transform=val_transform, is_train=False)
    test_loader = DataLoader(torch.utils.data.Subset(test_dataset, test_idx), batch_size=16, shuffle=False)
    
    fold_results = []
    print(f"Starting 5-Fold CV + Final Test on: {device}")

    # Initialize the highest record before the cross-validation loop
    best_overall_auc = 0.0

    # 2. Process 5-Fold Cross Validation (Internal Validation)
    for fold, (t_idx, v_idx) in enumerate(cv_splits):
        print(f"\n--- FOLD {fold+1}/5 ---")
        train_sub = torch.utils.data.Subset(GlaucomaDataset(csv_path, img_dir, transform=train_transform, is_train=True), train_val_idx[t_idx])
        val_sub = torch.utils.data.Subset(GlaucomaDataset(csv_path, img_dir, transform=val_transform, is_train=False), train_val_idx[v_idx])
        
        train_loader = DataLoader(train_sub, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=16, shuffle=False)
        
        model = GlaucomaEfficientNet(pretrained=True).to(device)
        criterion = WeightedQualityBCE(pos_weight=torch.tensor([0.363]).to(device))
        optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=15)
        
        best_fold_auc = 0.0
        for epoch in range(15):
            model.train()
            for imgs, labels, weights, _ in train_loader:
                imgs, labels, weights = imgs.to(device), labels.to(device).unsqueeze(1), weights.to(device).unsqueeze(1)
                optimizer.zero_grad(); loss = criterion(model(imgs), labels, weights); loss.backward(); optimizer.step()
            
            model.eval()
            v_preds, v_targets = [], []
            with torch.no_grad():
                for imgs, labels, _, _ in val_loader:
                    v_preds.extend(torch.sigmoid(model(imgs.to(device))).cpu().numpy())
                    v_targets.extend(labels.numpy())
            
            current_auc = roc_auc_score(v_targets, v_preds)
            
            if current_auc > best_fold_auc:
                best_fold_auc = current_auc
                
            # Save the model IF it breaks the overall best AUC record
            if current_auc > best_overall_auc:
                best_overall_auc = current_auc
                torch.save(model.state_dict(), 'best_model.pth')
                print(f"--> New record! Model saved with AUC: {current_auc:.4f}")
                
            scheduler.step()
        
        fold_results.append(best_fold_auc)
        print(f"Fold {fold+1} Result: {best_fold_auc:.4f}")

    # 3. FINAL EVALUATION (Final Blind Test)
    print("\n" + "="*40)
    print("PERFORMING FINAL EVALUATION ON BLIND TEST DATA")
    
    # Re-initialize clean model before loading weights
    model = GlaucomaEfficientNet(pretrained=False).to(device)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    test_probs, test_targets = [], []
    with torch.no_grad():
        for imgs, labels, _, _ in test_loader:
            test_probs.extend(torch.sigmoid(model(imgs.to(device))).cpu().numpy())
            test_targets.extend(labels.numpy())
    
    test_preds_bin = (np.array(test_probs) > 0.5).astype(int)
    
    print(f"Mean CV AUC: {np.mean(fold_results):.4f} +/- {np.std(fold_results):.4f}")
    print(f"FINAL TEST ROC-AUC: {roc_auc_score(test_targets, test_probs):.4f}")
    print(f"FINAL TEST F1-SCORE: {f1_score(test_targets, test_preds_bin):.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(test_targets, test_preds_bin)}")
    print("="*40)

if __name__ == '__main__':
    train_ultimate_pipeline()