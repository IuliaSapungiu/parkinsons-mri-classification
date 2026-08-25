import os
import pandas as pd
import numpy as np
import nibabel as nib

# --- FORCE HEADLESS BACKEND FOR HPC DEPLOYMENT ---
# Must be executed before importing pyplot to prevent X11 display connection errors
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from nilearn import datasets, image, plotting, regions

# ==========================================
# CONFIGURATION & SETSETTINGS
# ==========================================
TEST_MODE = False  # Turned OFF for full cohort cluster processing

# Standardized path matching your Stanage HPC folder hierarchy
PROJECT_ROOT = os.path.expanduser("~/dissertation")
INPUT_DIR = os.path.join(PROJECT_ROOT, "processed_data", "split_data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "processed_data", "extracted_features")
SPLITS = ["train", "val", "test"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Initializing ROI Pipeline (Intensity Features & Dissertation Graphics)...")
if TEST_MODE:
    print("!!! TEST MODE ACTIVE: Only processing 2 subjects per split !!!")

# --- 1. Load and ID the Harvard-Oxford Subcortical Atlas ---
print("\nLoading Harvard-Oxford Subcortical Atlas (1mm)...")
atlas = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-1mm')

if isinstance(atlas.maps, str):
    atlas_img = nib.load(atlas.maps)
else:
    atlas_img = atlas.maps

atlas_labels = atlas.labels
target_regions = ["Thalamus", "Caudate", "Putamen", "Pallidum", "Hippocampus", "Amygdala", "Accumbens"]

def find_label_index(region_name, side):
    search_term = f"{side} {region_name}".lower()
    for idx, label in enumerate(atlas_labels):
        if search_term in label.lower():
            return idx
    return None

# Generate and print the Appendix Table (IDs start at 4)
print("\n=======================================================")
print("DISSERTATION APPENDIX: HARVARD-OXFORD ATLAS ID MAPPING")
print("=======================================================")
print(f"{'Target Anatomical Structure':<30} | {'Atlas ID Value':<15}")
print("-" * 50)
for region in target_regions:
    idx_l = find_label_index(region, "Left")
    idx_r = find_label_index(region, "Right")
    print(f"Left {region:<25} | {str(idx_l):<15}")
    print(f"Right {region:<24} | {str(idx_r):<15}")
print("=======================================================\n")

LABEL_MAP = {
    0: "HC",   0.0: "HC",   "0": "HC",
    1: "PD",   1.0: "PD",   "1": "PD"
}

saved_hc_picture = False
saved_pd_picture = False

# --- 2. Feature Extraction Loop ---
for split in SPLITS:
    print(f"\n--- Processing {split.upper()} split ---")
    
    csv_path = os.path.join(INPUT_DIR, f"{split}_clinical.csv")
    if not os.path.exists(csv_path):
        print(f"[ERROR] Could not find clinical CSV at: {csv_path}")
        continue
        
    df = pd.read_csv(csv_path)
    
    if TEST_MODE:
        df = df.head(2)
        
    features_list = []
    
    for index, row in df.iterrows():
        try:
            subject_id = str(int(float(row['IDNUM'])))
        except (ValueError, TypeError):
            subject_id = str(row['IDNUM'])
        raw_label = row['PD_label']
        if pd.isna(raw_label) or raw_label not in LABEL_MAP:
            print(f"  [WARNING] Skipping Subject {subject_id}: Invalid/Missing PD_label")
            continue
        diagnosis = LABEL_MAP[raw_label]
        mri_path = os.path.join(INPUT_DIR, split, diagnosis, f"{subject_id}_reg.nii.gz")
        if not os.path.exists(mri_path):
            print(f"  [WARNING] File missing for Subject {subject_id}: {mri_path}")
            continue
            
        print(f"  Processing Subject: {subject_id} ({diagnosis})")
        pt_img = nib.load(mri_path)
        pt_data = pt_img.get_fdata()
        
        # --- Z-Score INTENSITY NORMALIZATION ---
        brain_mask = pt_data > 0
        brain_mean = np.mean(pt_data[brain_mask])
        brain_std = np.std(pt_data[brain_mask])
        norm_pt_data = (pt_data - brain_mean) / brain_std
        
        # Align Atlas
        resampled_atlas = image.resample_to_img(atlas_img, pt_img, interpolation='nearest')
        atlas_data = resampled_atlas.get_fdata()
        
        # --- 3. PERFECTED: AUTOMATED ILLUSTRATIVE VISUALIZATION FOR DISSERTATION ---
        if (diagnosis == "HC" and not saved_hc_picture) or (diagnosis == "PD" and not saved_pd_picture):
            pic_filename = f"dissertation_atlas_mapping_friendly_{diagnosis}.png"
            pic_path = os.path.join(OUTPUT_DIR, pic_filename)
            
            # Define the exact valid Target IDs from your appendix table
            valid_target_ids = [4, 5, 6, 7, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21]
            
            # Exact matrix matching using pure NumPy
            filtered_atlas_data = np.where(np.isin(atlas_data, valid_target_ids), atlas_data, 0)
            filter_atlas_img = nib.Nifti1Image(filtered_atlas_data, pt_img.affine, pt_img.header)
            
            print(f"  [VISUALIZATION] Generating friendly, labeled dissertation figure to: {pic_filename}")
            display = plotting.plot_roi(
                roi_img=filter_atlas_img,
                bg_img=pt_img,
                title=f"Subcortical ROI Segmentation Matrix with Illustrative Labels ({diagnosis})",
                output_file=None, 
                display_mode='ortho',
                cut_coords=(18, -8, -1), 
                colorbar=True
            )
            
            # --- 4. PERFECTED: ADD LABELS TO AXIAL VIEW (Z=-1) USING FRACTIONAL COORDINATES ---
            axial_plot = display.axes['z'].ax
            
            def add_friendly_label(ax, text, xy, xytext):
                """Draws a clean white arrow using relative coordinate fractions (0.0 to 1.0)."""
                ax.annotate(text, xy=xy, xytext=xytext, 
                            xycoords='axes fraction', textcoords='axes fraction',
                            arrowprops=dict(facecolor='white', edgecolor='white', width=1, headwidth=5, shrink=0.05),
                            fontsize=9, color='white', fontweight='bold',
                            horizontalalignment='center')

            # Precision-targeted fractional placements matching the default palette positions
            add_friendly_label(axial_plot, "Thalamus (L)", (0.44, 0.46), (0.22, 0.35))
            add_friendly_label(axial_plot, "Thalamus (R)", (0.56, 0.46), (0.78, 0.35))
            
            add_friendly_label(axial_plot, "Putamen (L)", (0.33, 0.53), (0.15, 0.65))
            add_friendly_label(axial_plot, "Putamen (R)", (0.67, 0.53), (0.85, 0.65))
            
            add_friendly_label(axial_plot, "Caudate (L)", (0.44, 0.61), (0.28, 0.85))
            add_friendly_label(axial_plot, "Caudate (R)", (0.56, 0.61), (0.72, 0.85))

            # Save the annotated layout cleanly
            plt.savefig(pic_path, bbox_inches='tight', dpi=300)
            plt.close()

            if diagnosis == "HC": saved_hc_picture = True
            if diagnosis == "PD": saved_pd_picture = True

        # --- 5. Extract Feature Metrics (With QC Alarm Integrated) ---
        pt_features = {'IDNUM': row['IDNUM']}
        for region in target_regions:
            idx_left = find_label_index(region, "Left")
            idx_right = find_label_index(region, "Right")
            
            voxels_left = norm_pt_data[atlas_data == idx_left]
            voxels_right = norm_pt_data[atlas_data == idx_right]
            
            if len(voxels_left) > 0:
                mean_left = np.mean(voxels_left)
                std_left = np.std(voxels_left)
            else:
                mean_left, std_left = 0.0, 0.0
                
            # QUALITY CONTROL CHECK: Trigger alarm for non-healthy Z-score
            if (abs(mean_left) > 2.5 and mean_left != 0.0) or (std_left == 0.0 and len(voxels_left) > 0):
                print(f"  [DATA QUALITY ALARM] Subject {subject_id} has highly anomalous Left {region} values! (Mean: {mean_left:.3f})")
            pt_features[f'{region}_Left_MeanIntensity'] = mean_left
            pt_features[f'{region}_Left_StdIntensity'] = std_left
            
            if len(voxels_right) > 0:
                mean_right = np.mean(voxels_right)
                std_right = np.std(voxels_right)
            else:
                mean_right, std_right = 0.0, 0.0
            if (abs(mean_right) > 2.5 and mean_right != 0.0) or (std_right == 0.0 and len(voxels_right) > 0):
                print(f"  [DATA QUALITY ALARM] Subject {subject_id} has highly anomalous Right {region} values! (Mean: {mean_right:.3f})")
            pt_features[f'{region}_Right_MeanIntensity'] = mean_right
            pt_features[f'{region}_Right_StdIntensity'] = std_right
            pt_features[f'{region}_Intensity_Asymmetry'] = mean_left - mean_right

        features_list.append(pt_features)
        
    # --- 6. Merge and export ---
    if not features_list:
        print(f"  [WARNING] No features extracted for {split} split. Skipping export.")
        continue
    features_df = pd.DataFrame(features_list)
    final_df = pd.merge(df, features_df, on='IDNUM', how='inner')
    output_filename = f"{split}_rf_features_TEST.csv" if TEST_MODE else f"{split}_rf_features.csv"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    final_df.to_csv(output_path, index=False)
    print(f"  -> Successfully saved features spreadsheet to: {output_path}")

print("\nExecution Finished Successfully!")