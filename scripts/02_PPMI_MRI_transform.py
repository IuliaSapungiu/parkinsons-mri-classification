"""
DICOM → NIfTI Conversion Script for PPMI Dataset
==================================================
Converts DICOM slices for each subject into a single 3D NIfTI volume (.nii.gz).
This is required before any CNN or ML analysis.

INPUT structure (from extract_ppmi_subjects.py):
    processed_data/
        PD/
            3000/
                sag_3D_FSPGR_BRAVO_straight/
                    2011-02-01_.../
                        I224562/
                            *.dcm
        HC/
            3002/ ...

OUTPUT structure:
    processed_data/
        nifti/
            PD/
                3000.nii.gz
                3005.nii.gz
                ...
            HC/
                3002.nii.gz
                ...
            conversion_log.csv   ← success/failure record for every subject

REQUIREMENTS (install once):
    pip install pydicom SimpleITK nibabel pandas numpy tqdm

    OR use dcm2niix (recommended, most robust):
        Download from: https://github.com/rordenlab/dcm2niix/releases
        Add to your system PATH, then set USE_DCM2NIIX = True below.
"""

import os
import subprocess
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import SimpleITK as sitk  

# ── Optional but recommended progress bar ──────────────────────────────────────
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(x, **kwargs): return x   # no-op fallback

# ─────────────────────────────────────────────
#  CONFIG  –  edit these before running
# ─────────────────────────────────────────────

PROCESSED_DATA_DIR = r"D:\UoS\1DISSERTATION\PD\processed_data"
OUTPUT_NIFTI_DIR   = r"D:\UoS\1DISSERTATION\PD\processed_data\nifti"

# ── Conversion method ──────────────────────────────────────────────────────────
# Option A (RECOMMENDED): dcm2niix — most reliable for PPMI/GE/Siemens data
#   Download: https://github.com/rordenlab/dcm2niix/releases
#   After downloading, either add to PATH or set DCM2NIIX_PATH below.
#
# Option B: Pure Python via SimpleITK — no extra install needed beyond pip
#   Slower and occasionally fails on complex DICOM headers, but works offline.

USE_DCM2NIIX   = True                 # set False to use SimpleITK instead
DCM2NIIX_PATH  = r"C:\Users\Administrator\Downloads\dcm2niix_win\dcm2niix.exe"

CLASSES = ["PD", "HC"]

# --- ABSOLUTE COHORT BLACKLIST ---
# These subjects have baseline MRI scans but lack mandatory clinical log records.
# Excluding them here ensures your downstream imaging data perfectly matches clinical tables.
BLACKLIST_IDS = ["3235", "4069", "5005", "5018"]

# ─────────────────────────────────────────────
#  LOGGING SETUP
# ─────────────────────────────────────────────

log_path = Path(OUTPUT_NIFTI_DIR) / "conversion.log"
Path(OUTPUT_NIFTI_DIR).mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  DICOM FINDER
# ─────────────────────────────────────────────

def find_dicom_dir(subject_folder: Path) -> Path | None:
    """
    Walk the subject folder tree and return the deepest directory
    that contains .dcm files. PPMI nests DICOMs several levels deep.
    """
    for root, dirs, files in os.walk(subject_folder):
        dcm_files = [f for f in files if f.lower().endswith(".dcm") or f.lower().endswith(".ima")]
        if dcm_files:
            return Path(root)
    return None


def count_dicoms(dicom_dir: Path) -> int:
    return sum(1 for f in dicom_dir.iterdir() if f.suffix.lower() in {".dcm", ".ima"})


# ─────────────────────────────────────────────
#  METHOD A: dcm2niix  (RECOMMENDED)
# ─────────────────────────────────────────────

def convert_dcm2niix(dicom_dir: Path, out_dir: Path, subject_id: str) -> bool:
    """
    Use dcm2niix to convert one subject's DICOM folder to NIfTI.
    Produces <subject_id>.nii.gz in out_dir.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        DCM2NIIX_PATH,
        "-z", "y",           # gzip compress → .nii.gz
        "-f", subject_id,    # output filename = subject ID
        "-o", str(out_dir),  # output directory
        "-m", "y",           # merge 2D slices into 3D volume
        str(dicom_dir)       # input DICOM folder
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            log.error(f"[{subject_id}] dcm2niix error:\n{result.stderr}")
            return False
        # Verify output was actually created
        outputs = list(out_dir.glob(f"{subject_id}*.nii.gz"))
        if not outputs:
            log.warning(f"[{subject_id}] dcm2niix ran but no .nii.gz found — check output manually.")
            return False
        # If multiple volumes were created (e.g. magnitude + phase), keep the largest
        if len(outputs) > 1:
            largest = max(outputs, key=lambda p: p.stat().st_size)
            for other in outputs:
                if other != largest:
                    other.unlink()
                    sidecar = other.with_suffix("").with_suffix(".json")
                    if sidecar.exists(): sidecar.unlink()
            largest.rename(out_dir / f"{subject_id}.nii.gz")
        return True
    except FileNotFoundError:
        log.error(
            "dcm2niix not found. Either:\n"
            "  1. Download from https://github.com/rordenlab/dcm2niix/releases and add to PATH\n"
            "  2. Set DCM2NIIX_PATH to the full path of the executable\n"
            "  3. Set USE_DCM2NIIX = False to use the SimpleITK fallback"
        )
        raise
    except subprocess.TimeoutExpired:
        log.error(f"[{subject_id}] dcm2niix timed out.")
        return False


# ─────────────────────────────────────────────
#  METHOD B: SimpleITK  (pure Python fallback)
# ─────────────────────────────────────────────

def convert_simpleitk(dicom_dir: Path, out_dir: Path, subject_id: str) -> bool:
    """
    Use SimpleITK to read a DICOM series and write a NIfTI volume.
    """
    try:
        import SimpleITK as sitk
    except ImportError:
        log.error("SimpleITK not installed. Run: pip install SimpleITK")
        raise

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{subject_id}.nii.gz"

    try:
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_dir))

        if not dicom_names:
            log.warning(f"[{subject_id}] SimpleITK found no DICOM series in {dicom_dir}")
            return False

        reader.SetFileNames(dicom_names)
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()
        image = reader.Execute()

        sitk.WriteImage(image, str(out_file))
        log.info(f"[{subject_id}] Written → {out_file}  size={image.GetSize()}")
        return True

    except Exception as e:
        log.error(f"[{subject_id}] SimpleITK failed: {e}")
        return False


# ─────────────────────────────────────────────
#  MAIN CONVERSION LOOP
# ─────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("DICOM → NIfTI Conversion")
    log.info(f"Method: {'dcm2niix' if USE_DCM2NIIX else 'SimpleITK'}")
    log.info("=" * 60)

    records = []   # for the CSV log

    for cls in CLASSES:
        class_input_dir  = Path(PROCESSED_DATA_DIR) / cls
        class_output_dir = Path(OUTPUT_NIFTI_DIR)   / cls
        class_output_dir.mkdir(parents=True, exist_ok=True)

        if not class_input_dir.exists():
            log.warning(f"Class folder not found, skipping: {class_input_dir}")
            continue

        subject_folders = sorted([p for p in class_input_dir.iterdir() if p.is_dir()])
        log.info(f"\n[{cls}] {len(subject_folders)} subjects found")

        for subj_folder in tqdm(subject_folders, desc=cls):
            subject_id = subj_folder.name
            out_nii    = class_output_dir / f"{subject_id}.nii.gz"
            # --- CRITICAL FIX: EXCLUDE COHORTS LACKING CLINICAL RECORDS ---
            if subject_id in BLACKLIST_IDS:
                log.info(f"[{subject_id}] Blacklisted — skipped due to missing baseline clinical logs.")
                records.append({"subject": subject_id, "group": cls, "status": "failed", "note": "excluded due to missing clinical logs"})
                continue

            # Skip if already converted
            if out_nii.exists():
                log.info(f"[{subject_id}] Already converted — skipping.")
                records.append({"subject": subject_id, "group": cls, "status": "skipped", "note": "already exists"})
                continue

            # Find the DICOM directory inside the subject folder
            dicom_dir = find_dicom_dir(subj_folder)
            if dicom_dir is None:
                log.warning(f"[{subject_id}] No DICOM files found under {subj_folder}")
                records.append({"subject": subject_id, "group": cls, "status": "failed", "note": "no DICOM files found"})
                continue

            n_slices = count_dicoms(dicom_dir)
            log.info(f"[{subject_id}] DICOM dir: {dicom_dir}  ({n_slices} slices)")

            # Convert
            if USE_DCM2NIIX:
                success = convert_dcm2niix(dicom_dir, class_output_dir, subject_id)
            else:
                success = convert_simpleitk(dicom_dir, class_output_dir, subject_id)

            status = "ok" if success else "failed"
            records.append({"subject": subject_id, "group": cls, "status": status,
                             "dicom_dir": str(dicom_dir), "n_slices": n_slices})

    # ── Summary ────────────────────────────────────────────────────────────────
    df_log = pd.DataFrame(records)
    log_csv = Path(OUTPUT_NIFTI_DIR) / "conversion_log.csv"
    df_log.to_csv(log_csv, index=False)

    ok      = (df_log["status"] == "ok").sum()
    skipped = (df_log["status"] == "skipped").sum()
    failed  = (df_log["status"] == "failed").sum()

    log.info("\n" + "=" * 60)
    log.info(f"Conversion complete.")
    log.info(f"  Converted : {ok}")
    log.info(f"  Skipped   : {skipped}  (already existed)")
    log.info(f"  Failed    : {failed}")
    log.info(f"  Log CSV   : {log_csv}")
    log.info("=" * 60)

    if failed > 0:
        log.warning("Some subjects failed — check conversion_log.csv for details.")


if __name__ == "__main__":
    main()