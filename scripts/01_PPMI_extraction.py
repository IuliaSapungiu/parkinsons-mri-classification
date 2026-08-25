"""
PPMI Subject Extraction & Alignment Script
==========================================
Selects 150 PD and 150 Healthy Control subjects from the PPMI dataset
and copies their DICOM files into a structured output directory.

HYBRID CAPABILITY:
- If target folders already exist and contain data, it bypasses the extraction 
  phase automatically, regenerates the manifest, and verifies against NIfTI logs.
- If target folders are empty, it runs a full fresh extraction for your supervisor.
"""

import os
import shutil
import random
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────
#  CONFIG  –  edit these before running
# ─────────────────────────────────────────────

CSV_PATH      = r"D:\UoS\1DISSERTATION\PD\raw\PPMI_Baseline_T1_MRI.csv"
PPMI_ROOT     = r"D:\UoS\1DISSERTATION\PD\raw\PPMI"     # folder that contains one sub-folder per subject ID
OUTPUT_DIR    = r"D:\UoS\1DISSERTATION\PD\processed_data"  # will be created if it doesn't exist
N_PER_CLASS   = 150          # how many subjects to select per class
RANDOM_SEED   = 42           # set to None for a different random pick each run
LOG_CSV_PATH  = r"D:\UoS\1DISSERTATION\PD\processed_data\nifti\conversion_log.csv"

SUBJECT_ID_COL = "Subject"   # e.g. 3000, 3001 …
DIAGNOSIS_COL  = "Group"     # values like "PD", "Control" / "HC" …

PD_LABEL      = "PD"         # exact string (case-sensitive)
HC_LABELS     = {"Control", "HC"}  # any of these = healthy

# ─────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────

def load_labels(csv_path: str) -> pd.DataFrame:
    """Load the CSV and return a tidy DataFrame with subject_id and group columns."""
    df = pd.read_csv(csv_path)
    print(f"[CSV] Loaded {len(df)} rows.  Columns: {list(df.columns)}")

    # Normalise column names (strip spaces)
    df.columns = df.columns.str.strip()

    required = {SUBJECT_ID_COL, DIAGNOSIS_COL}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing expected columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df[[SUBJECT_ID_COL, DIAGNOSIS_COL]].copy()
    df[SUBJECT_ID_COL] = df[SUBJECT_ID_COL].astype(str).str.strip()
    df[DIAGNOSIS_COL]  = df[DIAGNOSIS_COL].astype(str).str.strip()
    return df


def select_subjects(df: pd.DataFrame, n: int, seed) -> tuple[list, list]:
    """Return two lists: selected UNIQUE PD subject IDs, selected UNIQUE HC subject IDs."""
    rng = random.Random(seed)

    # CRITICAL FIX: Extract unique individuals from CSV before sampling to avoid duplicate rows
    pd_subjects = df[df[DIAGNOSIS_COL] == PD_LABEL][SUBJECT_ID_COL].unique().tolist()
    hc_subjects = df[df[DIAGNOSIS_COL].isin(HC_LABELS)][SUBJECT_ID_COL].unique().tolist()

    print(f"[CSV] Unique PD individuals available: {len(pd_subjects)}")
    print(f"[CSV] Unique HC individuals available: {len(hc_subjects)}")

    if len(pd_subjects) < n:
        raise ValueError(f"Not enough unique PD subjects ({len(pd_subjects)}) to select {n}.")
    if len(hc_subjects) < n:
        raise ValueError(f"Not enough unique HC subjects ({len(hc_subjects)}) to select {n}.")

    selected_pd = rng.sample(pd_subjects, n)
    selected_hc = rng.sample(hc_subjects, n)
    return selected_pd, selected_hc


def find_subject_folder(ppmi_root: str, subject_id: str) -> Path | None:
    """Return the Path to the subject's root folder inside PPMI_ROOT."""
    candidate = Path(ppmi_root) / subject_id
    if candidate.is_dir():
        return candidate

    matches = list(Path(ppmi_root).glob(f"*{subject_id}*"))
    if matches:
        return matches[0]

    return None


def copy_subject(src_folder: Path, dest_folder: Path):
    """Copy every file in src_folder recursively into dest_folder."""
    for src_file in src_folder.rglob("*"):
        if src_file.is_file():
            rel_path  = src_file.relative_to(src_folder)
            dest_file = dest_folder / rel_path
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dest_file)


def save_manifest(output_dir: str, pd_ids: list, hc_ids: list):
    """Save a simple text manifest containing only unique subjects."""
    manifest_path = Path(output_dir) / "selected_subjects.txt"
    with open(manifest_path, "w") as f:
        f.write(f"PD subjects ({len(pd_ids)}):\n")
        for sid in sorted(pd_ids, key=int):
            f.write(f"  {sid}\n")
        f.write(f"\nHC subjects ({len(hc_ids)}):\n")
        for sid in sorted(hc_ids, key=int):
            f.write(f"  {sid}\n")
    print(f"[Manifest] Saved unique subject list to {manifest_path}")


# ─────────────────────────────────────────────
#  MAIN HYBRID PIPELINE
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PPMI Subject Extraction & Verification Pipeline")
    print("=" * 60)

    # 1. Check if extraction has already occurred in the target directory
    pd_group_dir = Path(OUTPUT_DIR) / "PD"
    hc_group_dir = Path(OUTPUT_DIR) / "HC"
    
    already_extracted = False
    existing_pd_folders = []
    existing_hc_folders = []

    if pd_group_dir.is_dir() and hc_group_dir.is_dir():
        existing_pd_folders = [f.name for f in pd_group_dir.iterdir() if f.is_dir()]
        existing_hc_folders = [f.name for f in hc_group_dir.iterdir() if f.is_dir()]
        # If folders exist and contain subjects, trigger bypass safety check
        if len(existing_pd_folders) > 0 or len(existing_hc_folders) > 0:
            already_extracted = True

    # 2. Conditional Extraction Logic
    if already_extracted:
        print("[Already Extracted] Physical cohort folders detected in processed_data.")
        print("[*] Bypassing raw DICOM copy phase. Building manifest directly from existing directories...")
        pd_ids = sorted(existing_pd_folders, key=int)
        hc_ids = sorted(existing_hc_folders, key=int)
    else:
        print("[*] Fresh Run Detected. Preparing to execute full cohort selection and extraction...")
        # Load labels from master imaging tracker
        df = load_labels(CSV_PATH)
        # Select subjects strictly utilizing unique criteria
        pd_ids, hc_ids = select_subjects(df, N_PER_CLASS, RANDOM_SEED)
        print(f"\n[Selection] Chosen {len(pd_ids)} Distinct PD  |  {len(hc_ids)} Distinct HC")

    # 3. Create manifest text file (Ensures selected_subjects.txt is completely unique and accurate)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_manifest(OUTPUT_DIR, pd_ids, hc_ids)

    # 4. Physical Copy Execution (Only runs for your supervisor)
    if not already_extracted:
        groups = [("PD", pd_ids), ("HC", hc_ids)]
        for group_name, subject_ids in groups:
            group_dir = Path(OUTPUT_DIR) / group_name
            group_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n[Copy] → {group_dir}")

            not_found = []
            for i, sid in enumerate(subject_ids, 1):
                src = find_subject_folder(PPMI_ROOT, sid)
                if src is None:
                    print(f"  [{i:3d}/{N_PER_CLASS}] WARNING: folder not found for subject {sid}")
                    not_found.append(sid)
                    continue

                dest = group_dir / sid
                if dest.exists():
                    print(f"  [{i:3d}/{N_PER_CLASS}] SKIP (already copied): {sid}")
                    continue

                all_files = list(src.rglob("*"))
                n_files   = sum(1 for f in all_files if f.is_file())
                print(f"  [{i:3d}/{N_PER_CLASS}] Copying subject {sid}  ({n_files} files) ...", end=" ", flush=True)
                copy_subject(src, dest)
                print("done")

            if not_found:
                print(f"\n  [Warning] {len(not_found)} subjects not found in raw path: {not_found}")
    else:
        print(f"[Copy] Bypassed. {len(pd_ids)} PD and {len(hc_ids)} HC subjects loaded safely.")

    # 5. CROSS-VERIFICATION MODULE: Match Manifest vs NIfTI conversion_log.csv
    print("\n" + "-" * 50)
    print("Executing Validation Check: Manifest vs NIfTI Conversion Log")
    print("-" * 50)

    if not os.path.exists(LOG_CSV_PATH):
        print(f"[-] Cross-Verification Skipped: Log file not found at: {LOG_CSV_PATH}")
        print("    (This is normal if NIfTI conversion pipeline has not run yet.)")
    else:
        try:
            log_df = pd.read_csv(LOG_CSV_PATH)
            
            # Uncover subject identity column dynamically inside the conversion log
            log_id_col = None
            for col in log_df.columns:
                if col.strip().lower() in ['subject', 'patno', 'subject_id', 'id', 'subjectid']:
                    log_id_col = col
                    break
            
            if log_id_col is None:
                print("[-] Error: Unable to automatically identify Subject ID column in conversion_log.csv.")
                print(f"    Available columns: {list(log_df.columns)}")
            else:
                # Harmonize string types for clear matching evaluation
                log_subjects = set(log_df[log_id_col].astype(str).str.strip().unique())
                manifest_subjects = set(pd_ids + hc_ids)

                missing_in_log = manifest_subjects - log_subjects
                extra_in_log = log_subjects - manifest_subjects

                print(f"[Log Check] Found {len(log_subjects)} unique subjects tracked in conversion_log.csv.")
                
                if len(missing_in_log) == 0 and len(extra_in_log) == 0:
                    print("[✔] Success Validation! The manifest file and conversion_log.csv match up perfectly.")
                else:
                    if missing_in_log:
                        print(f"❌ Discrepancy Alert: {len(missing_in_log)} cohort subjects are MISSING from conversion_log.csv: {sorted(list(missing_in_log), key=int)}")
                    if extra_in_log:
                        print(f"⚠️ Information Notice: conversion_log.csv tracks {len(extra_in_log)} alternative subjects missing from this specific manifest.")
        
        except Exception as e:
            print(f"[-] Evaluation Error running check against conversion_log.csv: {e}")

    print("\n" + "=" * 60)
    print("Pipeline Execution Finished Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()