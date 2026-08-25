import os
import shutil
import glob
import pandas as pd

def extract_openneuro_t1():
    print("Starting OpenNeuro extraction process using processed CSV...")
    
    # 1. Define paths relative to your terminal location (/d/UoS/1DISSERTATION/PD)
    source_dir = 'ds001907'
    target_dir = 'openneuro'  # Now placed directly in the main PD folder
    csv_path = os.path.join('processed_data', 'openneuro_clinical_variables.csv')
    
    # 2. Check if your processed CSV exists
    if not os.path.exists(csv_path):
        print(f"Error: Could not find the processed CSV at {csv_path}")
        return

    # 3. Load the data
    print(f"Loading variables from {csv_path}...")
    df = pd.read_csv(csv_path)

    # 4. Create the target directories (openneuro/HC and openneuro/PD)
    for label in ['HC', 'PD']:
        os.makedirs(os.path.join(target_dir, label), exist_ok=True)
    print(f"Created target directories in: {os.path.abspath(target_dir)}")

    copied_count = 0
    missing_count = 0

    # 5. Loop through your processed dataframe
    for index, row in df.iterrows():
        # Get the ID and Label based on your specific column names
        raw_id = str(row['IDNUM']).strip()
        pd_label = row['PD_label']

        # Determine the group folder based on the binary label
        if pd_label == 1:
            group = 'PD'
        elif pd_label == 0:
            group = 'HC'
        else:
            print(f"Skipping ID {raw_id}: Unknown PD_label ({pd_label})")
            continue

        # Add the 'sub-' prefix to match the folder structure in ds001907
        subject_id = f"sub-{raw_id}"

        # Define the path to where the T1w scan should be (Session 1, Anatomical)
        anat_dir = os.path.join(source_dir, subject_id, 'ses-1', 'anat')
        
        if not os.path.exists(anat_dir):
            print(f"Missing anat folder for {subject_id}")
            missing_count += 1
            continue

        # Look for the T1-weighted .nii.gz file
        search_pattern = os.path.join(anat_dir, '*T1w*.nii.gz')
        t1_files = glob.glob(search_pattern)

        if t1_files:
            # Grab the first matching file
            source_file = t1_files[0]
            
            # Create a clean filename for the target directory
            clean_filename = f"{subject_id}_T1w.nii.gz"
            target_file = os.path.join(target_dir, group, clean_filename)
            
            # --- NEW ADDITION: Check if we already copied this file ---
            if os.path.exists(target_file):
                print(f"Skipping: {clean_filename} already exists in {group}/")
                # We still count it as a success for our summary
                copied_count += 1
                continue
            # ----------------------------------------------------------

            # Copy the file if it doesn't exist yet
            shutil.copy2(source_file, target_file)
            print(f"Copied: {clean_filename} -> {group}/")
            copied_count += 1
        else:
            print(f"No T1w scan found for {subject_id} in ses-1")
            missing_count += 1

    print("\n--- Extraction Summary ---")
    print(f"Successfully copied: {copied_count} files")
    print(f"Missing files: {missing_count}")
    print(f"Your clean dataset is ready at: {os.path.abspath(target_dir)}")

if __name__ == "__main__":
    extract_openneuro_t1()