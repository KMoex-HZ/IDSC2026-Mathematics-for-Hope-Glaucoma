import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
import torchvision.transforms as T
from torch.utils.data import DataLoader

train_transform = T.Compose([
    T.Resize((224, 224)),
    T.RandomRotation(degrees=15),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_train_test_splits(csv_path, test_size=0.2):
    df = pd.read_csv(csv_path)
    
    # Separate 20 percent of patients for a blind test set
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_val_idx, test_idx = next(gss.split(df, groups=df['Patient']))
    
    df_train_val = df.iloc[train_val_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)
    
    # Build 5 validation folds from the remaining 80 percent of training data
    gkf = GroupKFold(n_splits=5)
    cv_splits = list(gkf.split(df_train_val, groups=df_train_val['Patient']))
    
    return df, df_train_val, df_test, cv_splits, train_val_idx, test_idx