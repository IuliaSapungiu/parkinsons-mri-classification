import os
import pandas as pd
import numpy as np
import nibabel as nib
from pathlib import Path
from nilearn import datasets, image

# ==========================================
# CONFIGURATION
# ==========================================
print("Initializing OpenNeuro ROI Feature Extraction...")

PROJECT_ROOT = Path(".") # Assumes you run from D:\UoS\1DISSERTATION\PD
CLINICAL_CSV = PROJECT_ROOT / "processed_data" / "openneuro_clinical_variables.csv"
INPUT_NIFTI_DIR = PROJECT_ROOT / "processed_data" / "openneuro_registered"
OUTPUT_DIR = PROJECT_ROOT / "processed_data" / "openneuro_extracted_features"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP = {
    0: "HC",   0.0: "HC",   "0": "HC",
    1: "PD",   1.0: "PD",   "1": "PD"
}

# --- 1. Load Harvard-Oxford Subcortical Atlas ---
print("\nLoading Harvard-Oxford Subcortical Atlas (1mm)...")
atlas = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-1mm')

atlas_img = nib.load(atlas.maps) if isinstance(atlas.maps, str) else atlas.maps
atlas_labels = atlas.labels
target_regions = ["Thalamus", "Caudate", "Putamen", "Pallidum", "Hippocampus", "Amygdala", "Accumbens"]

def find_label_index(region_name, side):
    search_term = f"{side} {region_name}".lower()
    for idx, label in enumerate(atlas_labels):
        if search_term in label.lower():
            return idx
    return None

# --- 2. Load Clinical Data ---
print(f"\nReading demographics from: {CLINICAL_CSV.name}")
df = pd.read_csv(CLINICAL_CSV)

features_list = []

# --- 3. Feature Extraction Loop ---
print("\n--- Starting Extraction ---")
for index, row in df.iterrows():
    # Format the ID to match the OpenNeuro file naming convention
    subject_id = str(row['IDNUM']).strip()
    raw_label = row['PD_label']
    
    if pd.isna(raw_label) or raw_label not in LABEL_MAP:
        print(f"  [WARNING] Skipping Subject {subject_id}: Invalid/Missing PD_label")
        continue
        
    diagnosis = LABEL_MAP[raw_label]
    
    # Construct the exact file path: e.g., processed_data/openneuro_registered/HC/sub-RC4101_T1w.nii.gz
    filename = f"sub-{subject_id}_T1w.nii.gz"
    mri_path = INPUT_NIFTI_DIR / diagnosis / filename
    
    if not mri_path.exists():
        print(f"  [WARNING] NIfTI missing for {subject_id}: {filename}")
        continue
        
    print(f"  [➔] Processing: {subject_id} ({diagnosis})")
    pt_img = nib.load(str(mri_path))
    pt_data = pt_img.get_fdata()
    
    # --- Z-Score INTENSITY NORMALIZATION (Must match PPMI exactly) ---
    brain_mask = pt_data > 0
    brain_mean = np.mean(pt_data[brain_mask])
    brain_std = np.std(pt_data[brain_mask])
    norm_pt_data = (pt_data - brain_mean) / brain_std
    
    # Align Atlas to the registered brain
    resampled_atlas = image.resample_to_img(atlas_img, pt_img, interpolation='nearest')
    atlas_data = resampled_atlas.get_fdata()
    
    # --- Extract Feature Metrics ---
    pt_features = {'IDNUM': subject_id}
    
    for region in target_regions:
        idx_left = find_label_index(region, "Left")
        idx_right = find_label_index(region, "Right")
        
        voxels_left = norm_pt_data[atlas_data == idx_left]
        voxels_right = norm_pt_data[atlas_data == idx_right]
        
        # Left Hemisphere
        if len(voxels_left) > 0:
            mean_left, std_left = np.mean(voxels_left), np.std(voxels_left)
        else:
            mean_left, std_left = 0.0, 0.0
            
        pt_features[f'{region}_Left_MeanIntensity'] = mean_left
        pt_features[f'{region}_Left_StdIntensity'] = std_left
        
        # Right Hemisphere
        if len(voxels_right) > 0:
            mean_right, std_right = np.mean(voxels_right), np.std(voxels_right)
        else:
            mean_right, std_right = 0.0, 0.0
            
        pt_features[f'{region}_Right_MeanIntensity'] = mean_right
        pt_features[f'{region}_Right_StdIntensity'] = std_right
        
        # Asymmetry
        pt_features[f'{region}_Intensity_Asymmetry'] = mean_left - mean_right

    features_list.append(pt_features)

# --- 4. Merge and Export ---
print("\n--- Merging Data ---")
if features_list:
    features_df = pd.DataFrame(features_list)
    # Merge the clinical dataframe with our new MRI features based on IDNUM
    final_df = pd.merge(df, features_df, on='IDNUM', how='inner')
    
    output_path = OUTPUT_DIR / "openneuro_rf_features.csv"
    final_df.to_csv(output_path, index=False)
    print(f"[✓] Successfully saved all features to: {output_path}")
else:
    print("[ERROR] No features were extracted. Check your file paths.")

print("\nExecution Finished Successfully!")