import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, classification_report

print("=== INITIALIZING ULTIMATE SPATIAL-AWARE MULTIMODAL 3D CNN ===")

# ──────────────────────────────────────────────────────────────────────
# CONFIGURATION & PATHS
# ──────────────────────────────────────────────────────────────────────
DEBUG_MODE = False # Toggle to True for laptop testing, False for Stanage HPC

BASE_DIR = os.path.join("processed_data", "split_data")
RESULTS_DIR = os.path.join("results", "figures") 
MODEL_DIR = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train")
VAL_IMG_DIR = os.path.join(BASE_DIR, "val")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test")

TRAIN_CSV = os.path.join(BASE_DIR, "train_clinical.csv")
VAL_CSV = os.path.join(BASE_DIR, "val_clinical.csv")
TEST_CSV = os.path.join(BASE_DIR, "test_clinical.csv")

clinical_cols = [
    "age", "sex", "education_years", "family_history_pd",
    "moca_score", "gds_total", "verbal_fluency"
]

# Hyperparameters
EPOCHS = 2 if DEBUG_MODE else 50
BATCH_SIZE = 2 if DEBUG_MODE else 8  
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4  
EARLY_STOPPING_PATIENCE = 1 if DEBUG_MODE else 7  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using compute device: {device}")

# ──────────────────────────────────────────────────────────────────────
# 1. DATASET DEFINITION (INTEGRATED NORMALIZATION)
# ──────────────────────────────────────────────────────────────────────
class MultimodalParkinsonDataset(Dataset):
    def __init__(self, csv_path, split_dir, clinical_cols, imputer=None, is_train=False, debug=False):
        self.df = pd.read_csv(csv_path)
        if debug:
            self.df = self.df.head(4)
            
        self.split_dir = split_dir
        self.clinical_cols = clinical_cols
        
        if imputer is not None:
            if is_train:
                self.df[self.clinical_cols] = imputer.fit_transform(self.df[self.clinical_cols])
            else:
                self.df[self.clinical_cols] = imputer.transform(self.df[self.clinical_cols])
            
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        subject_id = str(int(row['IDNUM']))
        label = int(row['PD_label'])
        
        subfolder = "PD" if label == 1 else "HC"
        filename = f"{subject_id}_reg.nii.gz"
        img_path = os.path.join(self.split_dir, subfolder, filename)
        
        try:
            nib_img = nib.load(img_path)
            img_data = nib_img.get_fdata()
        except FileNotFoundError:
            print(f"CRITICAL ERROR: Cannot find image at {img_path}")
            raise
            
        # CLAUDE'S FIX: Robust Outlier Clipping & Z-score Normalization
        p_low  = np.percentile(img_data, 0.5)
        p_high = np.percentile(img_data, 99.5)
        img_data = np.clip(img_data, p_low, p_high)

        brain_mask = img_data > 0
        if brain_mask.sum() > 0:
            mu    = img_data[brain_mask].mean()
            sigma = img_data[brain_mask].std()
            img_data = (img_data - mu) / (sigma + 1e-8)
            
        img_data = np.expand_dims(img_data, axis=0) 
        clinical_vector = row[self.clinical_cols].values.astype(np.float32)
        
        return (
            torch.tensor(img_data, dtype=torch.float32), 
            torch.tensor(clinical_vector, dtype=torch.float32), 
            torch.tensor(label, dtype=torch.float32)
        )

# ──────────────────────────────────────────────────────────────────────
# 2. NEURAL NETWORK ARCHITECTURE (INTEGRATED SPATIAL AWARENESS)
# ──────────────────────────────────────────────────────────────────────
class MultimodalParkinsonCNN(nn.Module):
    def __init__(self, num_clinical_features=7):
        super(MultimodalParkinsonCNN, self).__init__()
        self.cnn_features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),
            # GEMINI'S FIX: Spatial quadrants (2,2,2) instead of flat (1,1,1)
            nn.AdaptiveAvgPool3d((2, 2, 2)) 
        )
        
        # 64 channels * 2 * 2 * 2 voxels = 512 dimensions + 7 clinical features = 519
        combined_dim = (64 * 2 * 2 * 2) + num_clinical_features
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(64, 1)
        )
        
    def forward(self, image, clinical_data):
        img_feats = self.cnn_features(image)
        img_feats = img_feats.view(img_feats.size(0), -1) 
        combined_feats = torch.cat((img_feats, clinical_data), dim=1) 
        logits = self.classifier(combined_feats)
        return logits.squeeze()

# ──────────────────────────────────────────────────────────────────────
# 3. MAIN TRAINING & EVALUATION LOOP
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n[1/5] Preparing Data Loaders...")
    imputer = SimpleImputer(strategy='median')
    
    train_dataset = MultimodalParkinsonDataset(TRAIN_CSV, TRAIN_IMG_DIR, clinical_cols, imputer=imputer, is_train=True, debug=DEBUG_MODE)
    val_dataset = MultimodalParkinsonDataset(VAL_CSV, VAL_IMG_DIR, clinical_cols, imputer=imputer, is_train=False, debug=DEBUG_MODE)
    test_dataset = MultimodalParkinsonDataset(TEST_CSV, TEST_IMG_DIR, clinical_cols, imputer=imputer, is_train=False, debug=DEBUG_MODE)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print("\n[2/5] Initializing Model & Advanced Optimizers...")
    model = MultimodalParkinsonCNN(num_clinical_features=len(clinical_cols)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = os.path.join(MODEL_DIR, "best_multimodal_3d_cnn.pth")
    
    print("\n[3/5] Beginning Training Phase...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        
        for images, clinical, labels in train_loader:
            images, clinical, labels = images.to(device), clinical.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images, clinical)
            if outputs.dim() == 0: outputs = outputs.unsqueeze(0)
                
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
                
        avg_train_loss = running_loss / len(train_loader)
        
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, clinical, labels in val_loader:
                images, clinical, labels = images.to(device), clinical.to(device), labels.to(device)
                outputs = model(images, clinical)
                if outputs.dim() == 0: outputs = outputs.unsqueeze(0)
                
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
        avg_val_loss = val_loss / len(val_loader)
        val_acc = (correct / total) * 100
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{EPOCHS}] ({epoch_time:.1f}s) | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_acc:.2f}%")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            if not DEBUG_MODE:
                print(f"   ➜ Validation loss improved. Saved optimal model to {best_model_path}")
        else:
            patience_counter += 1
            if not DEBUG_MODE:
                print(f"   ➜ No improvement in validation loss. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
            
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n[!] EARLY STOPPING TRIGGERED at Epoch {epoch+1}. Restoring best weights for testing.")
            break

    print("\n[4/5] Evaluating on Unseen Test Set...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, clinical, labels in test_loader:
            images, clinical, labels = images.to(device), clinical.to(device), labels.to(device)
            outputs = model(images, clinical)
            if outputs.dim() == 0: outputs = outputs.unsqueeze(0)
            
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    print("\nClassification Report (Test Set using Best Epoch Weights):")
    print(classification_report(all_labels, all_preds, target_names=['Healthy (0)', 'PD (1)']))

    print("\n[5/5] Generating Visualizations for Dissertation...")
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'], label='Val Loss', color='red')
    plt.title('Learning Curve (Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Accuracy', color='green')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "cnn_training_curves.png"), dpi=300)
    plt.close()
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=['Healthy (0)', 'PD (1)'],
                yticklabels=['Healthy (0)', 'PD (1)'])
    plt.title('3D CNN Test Set Confusion Matrix')
    plt.ylabel('Actual Diagnosis')
    plt.xlabel('Predicted Diagnosis')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "cnn_confusion_matrix.png"), dpi=300)
    plt.close()
    
    print(f"Visualizations saved to {RESULTS_DIR}")
    print("=== JOB COMPLETE ===")