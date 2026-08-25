import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_clinical_audit(final_df: pd.DataFrame, dataset_name="PPMI"):
    print(f"\n[*] Running Deep Statistical Audit on {dataset_name} Features...")
    
    clinical_bounds = {
        'age': (18, 110),
        'education_years': (0, 30),
        'updrs_motor_score': (0, 132),
        'moca_score': (0, 30),
        'gds_total': (0, 30),
        'tremor_score': (0, 40),
        'rigidity_score': (0, 20),
        'bradykinesia_score': (0, 44),
        'gait_instability_score': (0, 20)
    }

    audit_results = []
    for col in final_df.columns:
        if final_df[col].dtype in ['float64', 'int64'] and col not in ['IDNUM', 'PD_label']:
            col_min = final_df[col].min()
            col_max = final_df[col].max()
            col_mean = final_df[col].mean()
            
            status = "PASS"
            if col in clinical_bounds:
                valid_min, valid_max = clinical_bounds[col]
                if pd.notna(col_min) and col_min < valid_min:
                    status = "FAIL (Min Out of Bounds)"
                if pd.notna(col_max) and col_max > valid_max:
                    status = "FAIL (Max Out of Bounds)"
            
            audit_results.append({
                'Variable': col,
                'Min': round(col_min, 2) if pd.notna(col_min) else np.nan,
                'Max': round(col_max, 2) if pd.notna(col_max) else np.nan,
                'Mean': round(col_mean, 2) if pd.notna(col_mean) else np.nan,
                'Status': status
            })

    audit_df = pd.DataFrame(audit_results)
    print(f"\n--- {dataset_name} Clinical Distribution Summary ---")
    print(audit_df.to_string(index=False))

def main():
    print("================================================================")
    print("=== Step 01b: PPMI Clinical Harmonization Pipeline ===")
    print("================================================================")
    
    # =========================================================================
    # 1. SETUP BASE PROJECT DIRECTORY PATHS
    # =========================================================================
    current_dir = os.path.basename(os.getcwd())
    if current_dir == 'scripts':
        os.chdir('..')
        print("[*] Notice: Adjusted working directory back to project root folder.")

    # Input paths
    subjects_txt_path  = os.path.join('processed_data', 'selected_subjects.txt')
    ppmi_curated_path  = os.path.join('raw', 'PPMI_Curated_Data_Cut_Public_20251112.xlsx')
    ppmi_updrs3_path   = os.path.join('raw', 'MDS-UPDRS_Part_III_09Jun2026.csv')

    # Output paths
    output_csv_path    = os.path.join('processed_data', 'ppmi_harmonized_clinical_variables.csv')
    output_plot_dir    = os.path.join('results', 'figures')
    output_plot_path   = os.path.join(output_plot_dir, 'ppmi_clinical_correlation_matrix.png')
    
    os.makedirs(output_plot_dir, exist_ok=True)

    # =========================================================================
    # 2. PARSE CHOSEN PATIENT COHORTS FROM TXT
    # =========================================================================
    if not os.path.exists(subjects_txt_path):
        raise FileNotFoundError(f"[-] Critical Error: Cannot find subject list at {subjects_txt_path}")
        
    pd_subjects = []
    hc_subjects = []
    current_group = None

    with open(subjects_txt_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            if "PD subjects" in line:
                current_group = "PD"
                continue
            elif "HC subjects" in line:
                current_group = "HC"
                continue
            
            match = re.search(r'\d+', line)
            if match:
                sub_id = int(match.group())
                if current_group == "PD":
                    pd_subjects.append(sub_id)
                elif current_group == "HC":
                    hc_subjects.append(sub_id)

    all_selected_subjects = pd_subjects + hc_subjects
    print(f"[+] Loaded {len(all_selected_subjects)} cohorts from text file ({len(pd_subjects)} PD, {len(hc_subjects)} HC).")

    # =========================================================================
    # 3. LOAD & FILTER MASTER CURATED EXCEL SHEET (Baseline Only)
    # =========================================================================
    if not os.path.exists(ppmi_curated_path):
        raise FileNotFoundError(f"[-] Critical Error: Missing curated file at {ppmi_curated_path}")
        
    print("[+] Reading master clinical Excel sheet (this may take a moment)...")
    ppmi_curated_df = pd.read_excel(ppmi_curated_path) 

    id_col = None
    for alternative in ['PATNO', 'Subject', 'subject', 'patno']:
        if alternative in ppmi_curated_df.columns:
            id_col = alternative
            break
    if id_col is None:
        raise KeyError("[-] Error: Could not find an ID column in Excel sheet.")

    ppmi_curated_df[id_col] = pd.to_numeric(ppmi_curated_df[id_col], errors='coerce')
    
    if 'EVENT_ID' in ppmi_curated_df.columns:
        ppmi_curated_df = ppmi_curated_df[ppmi_curated_df['EVENT_ID'] == 'BL']
        print(f"[+] Curated Sheet: Successfully filtered for Baseline ('BL') visits.")

    filtered_curated_df = ppmi_curated_df[ppmi_curated_df[id_col].isin(all_selected_subjects)].copy()

    # =========================================================================
    # 4. LOAD, FILTER, & CALCULATE MDS-UPDRS MOTOR SUBSCORES (Baseline Only)
    # =========================================================================
    if not os.path.exists(ppmi_updrs3_path):
        raise FileNotFoundError(f"[-] Critical Error: Missing item-level UPDRS file at {ppmi_updrs3_path}")
        
    print("[+] Processing item-level MDS-UPDRS Part III CSV file...")
    updrs3_df = pd.read_csv(ppmi_updrs3_path, low_memory=False)
    updrs3_df['PATNO'] = pd.to_numeric(updrs3_df['PATNO'], errors='coerce')

    if 'EVENT_ID' in updrs3_df.columns:
        updrs3_df = updrs3_df[updrs3_df['EVENT_ID'] == 'BL']
        print(f"[+] MDS-UPDRS Sheet: Successfully filtered for Baseline ('BL') visits.")

    filtered_updrs3_df = updrs3_df[updrs3_df['PATNO'].isin(all_selected_subjects)].copy()

    rigidity_cols = ['NP3RIGN', 'NP3RIGRU', 'NP3RIGLU', 'NP3RIGRL', 'NP3RIGLL']
    tremor_cols   = ['NP3PTRMR', 'NP3PTRML', 'NP3KTRMR', 'NP3KTRML', 'NP3RTARU', 
                     'NP3RTALU', 'NP3RTARL', 'NP3RTALL', 'NP3RTALJ', 'NP3RTCON']
    brady_cols    = ['NP3FTAPR', 'NP3FTAPL', 'NP3HMOVR', 'NP3HMOVL', 'NP3PRSPR', 
                     'NP3PRSPL', 'NP3TTAPR', 'NP3TTAPL', 'NP3LGAGR', 'NP3LGAGL', 'NP3BRADY']
    gait_cols     = ['NP3GAIT', 'NP3FRZGT', 'NP3PSTBL', 'NP3POSTR', 'NP3RISNG']

    motor_groups = {
        'Tremor': tremor_cols,
        'Rigidity': rigidity_cols,
        'Bradykinesia': brady_cols,
        'Gait/Instability': gait_cols
    }
    
    print("[*] Diagnostic: Scanning individual motor items for placeholder anomalies (> 4)...")
    for group_name, cols in motor_groups.items():
        for col in cols:
            if col in filtered_updrs3_df.columns:
                filtered_updrs3_df[col] = pd.to_numeric(filtered_updrs3_df[col], errors='coerce')
                flagged_rows = filtered_updrs3_df[filtered_updrs3_df[col] > 4]
                if not flagged_rows.empty:
                    for idx, row in flagged_rows.iterrows():
                        print(f"  ⚠️ Placeholder Found: Patient {int(row['PATNO'])} has value '{row[col]}' in item '{col}' ({group_name} group)")
                
                # FOOLPROOF FIX: Force replacement using np.where to destroy anomalies completely
                filtered_updrs3_df[col] = np.where(filtered_updrs3_df[col] > 4, np.nan, filtered_updrs3_df[col])

    # Handle calculations safely
    filtered_updrs3_df['tremor_score'] = filtered_updrs3_df[tremor_cols].sum(axis=1, min_count=1)
    filtered_updrs3_df['rigidity_score'] = filtered_updrs3_df[rigidity_cols].sum(axis=1, min_count=1)
    filtered_updrs3_df['bradykinesia_score'] = filtered_updrs3_df[brady_cols].sum(axis=1, min_count=1)
    filtered_updrs3_df['gait_instability_score'] = filtered_updrs3_df[gait_cols].sum(axis=1, min_count=1)

    clean_subscores = filtered_updrs3_df[['PATNO', 'tremor_score', 'rigidity_score', 'bradykinesia_score', 'gait_instability_score']]
    
    # Audit for duplicates in UPDRS
    updrs_dupes = clean_subscores.duplicated(subset=['PATNO']).sum()
    print(f"[*] Audit: Found {updrs_dupes} duplicate subject entries in UPDRS file (retaining first).")
    clean_subscores = clean_subscores.drop_duplicates(subset=['PATNO'], keep='first')

    # =========================================================================
    # 5. MERGE DATASETS, HARMONIZE ENCODINGS, AND RENAME TO OPENNEURO SCHEMA
    # =========================================================================
    curated_targets = [id_col, 'COHORT', 'age_at_visit', 'SEX', 'EDUCYRS', 'updrs3_score', 'moca', 'gds', 'VLTANIM', 'duration_yrs', 'fampd_bin']
    available_curated_cols = [col for col in curated_targets if col in filtered_curated_df.columns]
    
    master_clinical_data = filtered_curated_df[available_curated_cols].copy()
    
    # Audit for duplicates in Curated
    curated_dupes = master_clinical_data.duplicated(subset=[id_col]).sum()
    print(f"[*] Audit: Found {curated_dupes} duplicate subject entries in Curated Master file (retaining first).")
    master_clinical_data = master_clinical_data.drop_duplicates(subset=[id_col], keep='first')

    # STRICT INNER MERGE: Drops any patient missing from the master clinical data
    master_template = pd.DataFrame({id_col: all_selected_subjects})
    final_df = master_template.merge(master_clinical_data, on=id_col, how='inner')
    
    if id_col == 'PATNO':
        final_df = final_df.merge(clean_subscores, on='PATNO', how='left')
    else:
        final_df = final_df.merge(clean_subscores, left_on=id_col, right_on='PATNO', how='left')
        if 'PATNO' in final_df.columns:
            final_df = final_df.drop(columns=['PATNO'])
    
    # RECODE COHORT MAPPING: Convert PPMI (1=PD, 2=HC) to Binary OpenNeuro (1=PD, 0=HC)
    final_df['COHORT'] = final_df['COHORT'].map({1: 1, 2: 0})

    # RENAME TO OPENNEURO SCHEMA
    rename_mapping = {
        id_col:          'IDNUM',
        'COHORT':        'PD_label',
        'age_at_visit':  'age',
        'SEX':           'sex',
        'EDUCYRS':       'education_years',
        'updrs3_score':  'updrs_motor_score',
        'moca':          'moca_score',
        'gds':           'gds_total',
        'VLTANIM':       'verbal_fluency',
        'duration_yrs':  'disease_duration_yrs',
        'fampd_bin':     'family_history_pd'
    }
    final_df = final_df.rename(columns=rename_mapping)

    # CRITICAL HARMONIZATION FIXES: Recode Family History and Clean up Float Decimals
    if 'family_history_pd' in final_df.columns:
        final_df['family_history_pd'] = final_df['family_history_pd'].map({1: 1, 2: 0})
    
    if 'age' in final_df.columns:
        final_df['age'] = final_df['age'].round(2)
        
    if 'disease_duration_yrs' in final_df.columns:
        final_df['disease_duration_yrs'] = final_df['disease_duration_yrs'].round(2)

    # Reorder columns
    ideal_order = ['IDNUM', 'PD_label', 'age', 'sex', 'education_years', 'updrs_motor_score', 
                   'tremor_score', 'rigidity_score', 'bradykinesia_score', 'gait_instability_score', 
                   'moca_score', 'gds_total', 'verbal_fluency', 'disease_duration_yrs', 'family_history_pd']
    existing_order = [c for c in ideal_order if c in final_df.columns]
    final_df = final_df[existing_order]

    # Export clean harmonized validation matrix
    final_df.to_csv(output_csv_path, index=False)
    print(f"\n[+] Success! Cleansed baseline cohort matrix saved to: {output_csv_path}")
    print(f"[+] Verified Matrix Shape: {final_df.shape[0]} rows (subjects) by {final_df.shape[1]} clinical variables.")

    # Call the audit right here!
    run_clinical_audit(final_df, dataset_name="PPMI")

    # Missing reporting (Updated to reflect exact drop mechanics)
    curated_ids = set(filtered_curated_df[id_col].dropna().astype(int))
    missing_ids = set(all_selected_subjects) - curated_ids
    if missing_ids:
        print(f"\n[-] Explanation: The following {len(missing_ids)} subjects had no documented clinic baseline (BL) logs inside the curated file:")
        print(f"    {sorted(list(missing_ids))}")
        print("    [!] FIX ENFORCED: Dropped via strict 'inner' mapping. The final cohort is rigorously aligned to valid MRI subjects.")

    # =========================================================================
    # 6. GENERATE UPDATED DIAGNOSTIC CORRELATION MATRIX
    # =========================================================================
    print("\n[+] Plotting updated diagnostic clinical heatmap...")
    cols_for_corr = [c for c in final_df.select_dtypes(include=[np.number]).columns if c not in ['IDNUM']]

    if len(cols_for_corr) > 1:
        corr_matrix = final_df[cols_for_corr].corr(method='pearson')
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Correlation Matrix: Fully Harmonized PPMI Clinical Schema (Baseline Cohort)', fontsize=12, pad=15)
        plt.tight_layout()
        plt.savefig(output_plot_path, dpi=300)
        print(f"[+] Heatmap graphic exported directly to: {output_plot_path}")
        
    print("================================================================")

if __name__ == '__main__':
    main()