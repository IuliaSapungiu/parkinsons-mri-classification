import os
import subprocess
import logging
import random
from pathlib import Path
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs): return x

# ─────────────────────────────────────────────
#  CONFIG  –  edit these before running
# ─────────────────────────────────────────────

INPUT_NIFTI_DIR     = Path(r"D:\UoS\1DISSERTATION\PD\openneuro") 
OUTPUT_STRIPPED_DIR = Path(r"D:\UoS\1DISSERTATION\PD\processed_data\openneuro_skull_stripped")

CLASSES = ["HC", "PD"]

# --- TEST MODE SETTINGS ---
TEST_MODE = False   
TEST_SAMPLE_SIZE = 1  

# --- HD-BET Settings ---
DEVICE = "cpu"      
        
# ─────────────────────────────────────────────
#  LOGGING SETUP
# ─────────────────────────────────────────────

Path(OUTPUT_STRIPPED_DIR).mkdir(parents=True, exist_ok=True)
log_path = Path(OUTPUT_STRIPPED_DIR) / "openneuro_skull_strip.log"

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
#  MAIN PROCESSING LOOP
# ─────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("OpenNeuro Brain Extraction using HD-BET")
    log.info(f"Input Directory : {INPUT_NIFTI_DIR}")
    log.info(f"Output Directory: {OUTPUT_STRIPPED_DIR}")
    log.info(f"Compute Device  : {DEVICE}")
    log.info(f"Test Mode       : {'ON (Picking ' + str(TEST_SAMPLE_SIZE) + ' random subjects)' if TEST_MODE else 'OFF (Processing ALL)'}")
    log.info("=" * 60)

    records = []

    for cls in CLASSES:
        class_input_dir  = INPUT_NIFTI_DIR / cls
        class_output_dir = OUTPUT_STRIPPED_DIR / cls
        
        # Create output directories if they don't exist
        class_output_dir.mkdir(parents=True, exist_ok=True)

        if not class_input_dir.exists():
            log.warning(f"Class folder not found, skipping: {class_input_dir}")
            continue

        # Find all .nii.gz files
        nifti_files = sorted(list(class_input_dir.glob("*.nii.gz")))
        
        # Apply Test Mode Subsampling
        if TEST_MODE:
            random.seed(42)
            sample_size = min(TEST_SAMPLE_SIZE, len(nifti_files))
            nifti_files = random.sample(nifti_files, sample_size)
            
        log.info(f"\n[{cls}] Processing {len(nifti_files)} subjects...")

        for in_nii in tqdm(nifti_files, desc=cls):
            subject_filename = in_nii.name
            out_nii = class_output_dir / subject_filename

            # Skip if already processed
            if out_nii.exists():
                log.info(f"[{subject_filename}] Already skull-stripped — skipping.")
                records.append({"subject": subject_filename, "group": cls, "status": "skipped"})
                continue

            # Construct the HD-BET command line argument
            cmd = [
                "hd-bet", 
                "-i", str(in_nii), 
                "-o", str(out_nii), 
                "-device", DEVICE, 
                "--disable_tta"  
            ]

            log.info(f"[{subject_filename}] Running HD-BET...")
            
            try:
                # Run the command
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0 and out_nii.exists():
                    log.info(f"[{subject_filename}] Success!")
                    records.append({"subject": subject_filename, "group": cls, "status": "ok"})
                else:
                    log.error(f"[{subject_filename}] HD-BET failed. Error: {result.stderr}")
                    records.append({"subject": subject_filename, "group": cls, "status": "failed"})

            except FileNotFoundError:
                log.error("HD-BET command not found. Ensure it is installed in your current environment.")
                return

    # ── Summary ────────────────────────────────────────────────────────────────
    if records:
        df_log = pd.DataFrame(records)
        log_csv = OUTPUT_STRIPPED_DIR / "openneuro_skull_strip_log.csv"
        df_log.to_csv(log_csv, index=False)

        ok      = (df_log["status"] == "ok").sum()
        skipped = (df_log["status"] == "skipped").sum()
        failed  = (df_log["status"] == "failed").sum()

        log.info("\n" + "=" * 60)
        log.info(f"OpenNeuro Skull Stripping complete.")
        log.info(f"  Successfully Stripped : {ok}")
        log.info(f"  Skipped (Pre-existing): {skipped}")
        log.info(f"  Failed                : {failed}")
        log.info(f"  Log CSV               : {log_csv}")
        log.info("=" * 60)

if __name__ == "__main__":
    main()