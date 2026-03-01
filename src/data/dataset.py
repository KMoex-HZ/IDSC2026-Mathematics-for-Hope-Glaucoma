import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class GlaucomaDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, is_train=True):
        self.img_dir = img_dir
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.is_train = is_train
        
        # Map text labels to binary numerical values
        self.df['Label_Num'] = self.df['Label'].map({'GON+': 1, 'GON-': 0})
        
        # Normalize Quality Score into mathematical weights (0.1 to 1.0)
        self.df['Quality_Weight'] = self.df['Quality Score'] / 10.0

    def __len__(self):
        return len(self.df)

    def apply_clahe(self, image_bgr):
        # Contrast extraction without distorting original colors using LAB color space
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        
        limg = cv2.merge((cl,a,b))
        image_rgb = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return image_rgb

    def crop_eye(self, image_rgb):
        # Detect eyeball contours and crop black margins to save memory
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            return image_rgb[y:y+h, x:x+w]
        return image_rgb

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['Image Name']
        label = row['Label_Num']
        weight = row['Quality_Weight']
        patient_id = row['Patient']
        
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        
        # Extensive preprocessing before feeding into the Deep Learning architecture
        image = self.apply_clahe(image)
        image = self.crop_eye(image)
        
        # Convert OpenCV array to PIL image object
        image_pil = Image.fromarray(image)
        
        if self.transform:
            image_pil = self.transform(image_pil)
            
        return image_pil, torch.tensor(label, dtype=torch.float32), torch.tensor(weight, dtype=torch.float32), patient_id