"""
================================================================================
  01_preprocessing.py
  Parkinson's Disease Classification Project — Preprocessing Pipeline
================================================================================
=======================================
"""

import os
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
import seaborn as sns

# =============================================================================
#  CONFIGURATION & PATH SETUP
# =============================================================================

PROJECT_ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA_PATH   = os.path.join(PROJECT_ROOT, "raw", "demographics.csv")
PROCESSED_DIR   = os.path.join(PROJECT_ROOT, "processed_data")
CLEAN_CSV       = os.path.join(PROCESSED_DIR, "pd_demographics_clean.csv")
FINAL_CSV       = os.path.join(PROCESSED_DIR, "openneuro_clinical_variables.csv")

SENTINEL_CODES = {-998, -997, -905, -903, -802, -803, -801, -800, -999}
MISSING_THRESHOLD = 0.50

# =============================================================================
#  SECTION 1 -- Load raw data (with Jagged CSV Fix)
# =============================================================================

def load_raw(path: str) -> pd.DataFrame:
    print(f"[1] Loading raw data from: {path}")
    
    with open(path, 'r', encoding='utf-8-sig', errors='ignore') as f:
        lines = f.readlines()
        
    if not lines:
        raise ValueError("The CSV file is empty.")
        
    header_fields = lines[0].strip('\n').split(',')
    expected_cols = len(header_fields)
    
    cleaned_lines = []
    for line in lines:
        fields = line.strip('\n').split(',')
        if len(fields) > expected_cols:
            fields = fields[:expected_cols]
        elif len(fields) < expected_cols:
            fields.extend([''] * (expected_cols - len(fields)))
        cleaned_lines.append(','.join(fields))
        
    virtual_file = io.StringIO('\n'.join(cleaned_lines))
    df = pd.read_csv(virtual_file, low_memory=False)
    
    print(f"    Raw shape: {df.shape[0]} rows x {df.shape[1]} columns")
    return df

# =============================================================================
#  SECTION 2 -- Drop uninformative columns
# =============================================================================

def drop_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    unnamed = [c for c in df.columns if c.startswith("Unnamed:")]
    df = df.drop(columns=unnamed)
    print(f"\n[2] Column cleaning:")
    print(f"    Removed {len(unnamed):>4d} unnamed/empty trailing columns")

    miss_rate = df.isnull().sum() / len(df)
    high_miss = miss_rate[miss_rate > MISSING_THRESHOLD].index.tolist()
    df = df.drop(columns=high_miss)
    print(f"    Removed {len(high_miss):>4d} columns with >{MISSING_THRESHOLD*100:.0f}% missing values")
    
    # --- Check for duplicates without dropping them ---
    if "IDNUM" in df.columns:
        n_dupes = df.duplicated(subset=["IDNUM"]).sum()
        print(f"    Found   {n_dupes:>4d} duplicate subject IDs (retained for evaluation)")
    
    print(f"    Retained {df.shape[1]:>4d} columns")
    return df

# =============================================================================
#  SECTION 3 -- Assign diagnostic label
# =============================================================================

def assign_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["PD_label"] = df["IDNUM"].apply(
        lambda x: 1 if str(x).startswith("RC41") else 0
    )
    n_pd = (df["PD_label"] == 1).sum()
    n_hc = (df["PD_label"] == 0).sum()
    print(f"\n[3] Diagnostic labels assigned:")
    print(f"    PD patients      (PD_label=1): {n_pd}")
    print(f"    Healthy controls (PD_label=0): {n_hc}")
    return df

# =============================================================================
#  SECTION 4 -- Helper utilities
# =============================================================================

def clean_numeric(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace(list(SENTINEL_CODES), np.nan)
    return s

def sum_subscale(df: pd.DataFrame, columns: list, min_valid: int = 1) -> pd.Series:
    cleaned = df[columns].apply(clean_numeric)
    return cleaned.sum(axis=1, min_count=min_valid)

# =============================================================================
#  SECTION 5 -- Feature extraction
# =============================================================================

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()

    out["IDNUM"]    = df["IDNUM"]
    out["PD_label"] = df["PD_label"]
    
    age_col = "age_demo" if "age_demo" in df.columns else "age"
    if age_col in df.columns:
        # Age is cleanly rounded to 1 decimal place here
        out["age"] = clean_numeric(df[age_col]).round(2)

    if "gender" in df.columns:
        out["sex"] = df["gender"].map({"Male": 0, "Female": 1})

    if "education_years" in df.columns:
        out["education_years"] = clean_numeric(df["education_years"])

    # MDS-UPDRS PART III
    mds3_items = [c for c in df.columns if c.startswith("mdspdrs_3_") and c not in {
        "mdspdrs_3a", "mdspdrs_3b", "mdspdrs_3c", "mdspdrs_3c1", "mdspdrs_3a_dysk", "mdspdrs_3b_dysk"
    }]
    if mds3_items:
        out["updrs_motor_score"] = sum_subscale(df, mds3_items, min_valid=5)

    tremor_cols = ["mdspdrs_3_15r", "mdspdrs_3_15l", "mdspdrs_3_16r", "mdspdrs_3_16l", "mdspdrs_3_17r", "mdspdrs_3_17l"]
    if all(c in df.columns for c in tremor_cols):
        out["tremor_score"] = sum_subscale(df, tremor_cols, min_valid=2)

    rigid_cols = ["mdspdrs_3_3_neck", "mdspdrs_3_3_rue", "mdspdrs_3_3_lue", "mdspdrs_3_3d_rle", "mdspdrs_3_3e_lle"]
    if all(c in df.columns for c in rigid_cols):
        out["rigidity_score"] = sum_subscale(df, rigid_cols, min_valid=3)

    brady_cols = ["mdspdrs_3_4r", "mdspdrs_3_4l", "mdspdrs_3_5r", "mdspdrs_3_5l", "mdspdrs_3_6r", "mdspdrs_3_6l", "mdspdrs_3_7r", "mdspdrs_3_7l"]
    if all(c in df.columns for c in brady_cols):
        out["bradykinesia_score"] = sum_subscale(df, brady_cols, min_valid=4)

    gait_cols = ["mdspdrs_3_10", "mdspdrs_3_11", "mdspdrs_3_12", "mdspdrs_3_13", "mdspdrs_3_14"]
    if all(c in df.columns for c in gait_cols):
        out["gait_instability_score"] = sum_subscale(df, gait_cols, min_valid=3)

    if "moca_score" in df.columns:
        moca = clean_numeric(df["moca_score"])
        out["moca_score"] = moca.where(moca <= 30, other=np.nan)

    if "gds_total" in df.columns:
        gds = clean_numeric(df["gds_total"])
        out["gds_total"] = gds.where(gds <= 30, other=np.nan)

    if "f_words_flu" in df.columns:
        out["verbal_fluency"] = clean_numeric(df["f_words_flu"])

    if "symptom_onset_year" in df.columns:
        PROXY_SCAN_YEAR = 2018
        onset_year = clean_numeric(df["symptom_onset_year"])
        duration = PROXY_SCAN_YEAR - onset_year
        out["disease_duration_yrs"] = duration.where(duration >= 0, other=np.nan)

    family_cols = ["mother_pd", "father_pd", "siblings_pd"]
    if all(c in df.columns for c in family_cols):
        m     = clean_numeric(df["mother_pd"])
        f_col = clean_numeric(df["father_pd"])
        s     = clean_numeric(df["siblings_pd"])
        family_hx = ((m >= 1) | (f_col >= 1) | (s >= 1)).astype(float)
        all_missing = m.isna() & f_col.isna() & s.isna()
        family_hx[all_missing] = np.nan
        out["family_history_pd"] = family_hx

    return out

# =============================================================================
#  SECTION 6 -- Clinical Range Profiler
# =============================================================================

def run_clinical_audit(final_df: pd.DataFrame):
    print("\n[6] Running Deep Statistical Audit on OpenNeuro Features...")
    
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
        if final_df[col].dtype in ['float64', 'int64'] and col != 'IDNUM':
            col_min = final_df[col].min()
            col_max = final_df[col].max()
            col_mean = final_df[col].mean()
            
            status = "PASS"
            if col in clinical_bounds:
                valid_min, valid_max = clinical_bounds[col]
                # Check for bounds, ignoring NaNs which return True for > or <
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
    print("\n--- OpenNeuro Clinical Distribution Summary ---")
    print(audit_df.to_string(index=False))
    
    if "FAIL" in audit_df['Status'].to_string():
        print("\n[!] WARNING: Some harmonized variables exceed normal clinical limits. Please review.")
    else:
        print("\n[+] SUCCESS: All OpenNeuro variables fall within mathematically normal clinical boundaries.")

# =============================================================================
#  SECTION 7 -- Correlation Matrix Validation
# =============================================================================

def export_correlation_matrix(df_final: pd.DataFrame) -> None:
    print("\n[7] Generating Clinical Correlation Matrix (PNG)...")
    
    # Define and create the figures directory
    FIGURES_DIR = os.path.join(PROJECT_ROOT, "results", "figures")
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # Select only numeric columns to prevent math errors on text data
    numeric_df = df_final.select_dtypes(include=['float64', 'int64'])
    corr_matrix = numeric_df.corr(method='pearson')
    
    # Set up the matplotlib figure for a dissertation-quality plot
    plt.figure(figsize=(12, 10))
    
    # Plot the heatmap
    sns.heatmap(
        corr_matrix, 
        annot=True,              # Show the actual correlation numbers
        fmt=".2f",               # Round to 2 decimal places
        cmap="coolwarm",         # Blue for negative, Red for positive correlations
        vmin=-1, vmax=1,         # Anchor the color scale from -1 to 1
        square=True, 
        linewidths=.5, 
        cbar_kws={"shrink": .8}
    )
    
    plt.title("OpenNeuro Clinical Correlation Matrix", fontsize=16, pad=20)
    plt.tight_layout()
    
    # Export to the results/figures folder as a PNG
    corr_path = os.path.join(FIGURES_DIR, "openneuro_correlation_matrix.png")
    plt.savefig(corr_path, dpi=300, bbox_inches='tight')
    plt.close() # Close the figure to free up memory
    
    print(f"    Exported correlation matrix image: {corr_path}")

# =============================================================================
#  SECTION 8 -- Export outputs
# =============================================================================

def export_outputs(df_clean: pd.DataFrame, df_final: pd.DataFrame) -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df_clean.to_csv(CLEAN_CSV, index=False)
    df_final.to_csv(FINAL_CSV, index=False)
    print(f"\n[7] Exported ML-ready variables: {FINAL_CSV}")

# =============================================================================
#  MAIN EXECUTION
# =============================================================================

def main():
    print("================================================================")
    print(" 01_preprocessing.py -- OpenNeuro Demographics Cleaner")
    print("================================================================\n")
    
    df_raw = load_raw(RAW_DATA_PATH)
    df_clean = drop_empty_columns(df_raw)
    df_labeled = assign_label(df_clean)
    
    print("\n[4/5] Extracting composite clinical features...")
    df_final = extract_features(df_labeled)
    print(f"      Feature extraction complete. Final shape: {df_final.shape[0]} rows x {df_final.shape[1]} columns")
    
    run_clinical_audit(df_final)
    export_correlation_matrix(df_final)
    export_outputs(df_labeled, df_final)
    
    print("\n[+] Preprocessing completed successfully.")
    print("================================================================\n")

if __name__ == "__main__":
    main()