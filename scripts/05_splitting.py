import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# =====================================================================
# 1. PATH CONFIGURATIONS (RELATIVE TO THE ROOT 'dissertation' FOLDER)
# =====================================================================
csv_path = "processed_data/ppmi_harmonized_clinical_variables.csv"
mri_base_dir = "processed_data/registration"
output_base_dir = "processed_data/split_data"

# Mapping column values to actual folder names
# 1 = Parkinson's Disease (PD), 0 = Healthy Control (HC)
label_to_folder = {1: "PD", 0: "HC"}

# =====================================================================
# 2. CREATE BALANCED DESTINATION DIRECTORIES
# =====================================================================
print("Initializing target directory structure...")
for split in ["train", "val", "test"]:
    for folder_name in label_to_folder.values():
        os.makedirs(os.path.join(output_base_dir, split, folder_name), exist_ok=True)

# =====================================================================
# 3. LOAD AND STRATIFY CLINICAL SPREADSHEET
# =====================================================================
print(f"Loading master clinical matrix: {csv_path}")
df = pd.read_csv(csv_path)

# Enforce string types for ID matching to protect any leading zeros
df["IDNUM"] = df["IDNUM"].astype(str)

print("Executing stratified train/val/test partitioning (70/15/15)...")
# Split 1: 70% Train, 30% Temporary (to be split into Val and Test)
train_df, temp_df = train_test_split(
    df, 
    test_size=0.30, 
    random_state=42,       # Seed locked for absolute replication
    stratify=df["PD_label"] # Ensures identical group ratios across subsets
)

# Split 2: Divide the 30% remaining perfectly in half (15% Val, 15% Test)
val_df, test_df = train_test_split(
    temp_df, 
    test_size=0.50, 
    random_state=42, 
    stratify=temp_df["PD_label"]
)

# Save the subset CSVs for future Machine Learning pipelines
train_df.to_csv(os.path.join(output_base_dir, "train_clinical.csv"), index=False)
val_df.to_csv(os.path.join(output_base_dir, "val_clinical.csv"), index=False)
test_df.to_csv(os.path.join(output_base_dir, "test_clinical.csv"), index=False)

print("\n--- Partitioning Breakdown ---")
print(f"  Total Valid Cohort : {len(df)} subjects")
print(f"  Training Set (70%) : {len(train_df)} subjects")
print(f"  Validation Set (15%): {len(val_df)} subjects")
print(f"  Testing Set (15%)  : {len(test_df)} subjects\n")

# =====================================================================
# 4. SORT AND COPY REGISTERED MRI SCANS
# =====================================================================
def dispatch_mri_files(dataframe, split_name):
    print(f"Sorting physical MRI volumes into '{split_name}' directories...")
    success_count = 0
    missing_count = 0
    
    for _, row in dataframe.iterrows():
        subject_id = row["IDNUM"]
        label = int(row["PD_label"])
        group_folder = label_to_folder[label] # Evaluates to 'PD' or 'HC'
        
        # Define file target coordinates
        filename = f"{subject_id}_reg.nii.gz"
        source_path = os.path.join(mri_base_dir, group_folder, filename)
        dest_path = os.path.join(output_base_dir, split_name, group_folder, filename)
        
        # Transfer file safely if found on disk
        if os.path.exists(source_path):
            shutil.copy2(source_path, dest_path) # copy2 retains original timestamps
            success_count += 1
        else:
            missing_count += 1
            
    print(f"  -> Finished {split_name}: {success_count} moved successfully. ({missing_count} missing)")

# Run sorting operations
dispatch_mri_files(train_df, "train")
dispatch_mri_files(val_df, "val")
dispatch_mri_files(test_df, "test")

print("\nDataset split complete. Tabular and structural boundaries locked down.")