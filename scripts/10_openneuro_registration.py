import ants
import os
from pathlib import Path

def main():
    print("=" * 60)
    print("HPC BATCH RUN: OpenNeuro SyN Registration (All Subjects)")
    print("=" * 60)

    # 1. Define Relative Paths
    INPUT_BASE  = Path("processed_data/openneuro_skull_stripped")
    OUTPUT_BASE = Path("processed_data/openneuro_registered")
    CLASSES     = ["HC", "PD"]

    # 2. Load the standard template once to save memory
    print("Loading standard MNI template via ANTs...")
    fixed_img = ants.image_read(ants.get_ants_data('mni'))

    for cls in CLASSES:
        class_in_dir = INPUT_BASE / cls
        class_out_dir = OUTPUT_BASE / cls

        # Create output directories if they don't exist
        class_out_dir.mkdir(parents=True, exist_ok=True)

        if not class_in_dir.exists():
            print(f"[ERROR] Path missing: {class_in_dir}")
            continue

        # Get all skull-stripped files in this folder
        all_files = sorted(list(class_in_dir.glob("*.nii.gz")))

        if not all_files:
            print(f"No files found in {class_in_dir}")
            continue

        print(f"\nProcessing [{cls}] cohort: {len(all_files)} brains found.")

        # 3. CRITICAL CHANGE: Loop through ALL files in the list
        for target_file in all_files:
            output_file = class_out_dir / target_file.name

            # Skip if the file has already been processed!
            if output_file.exists():
                print(f"  [SKIPPING] {target_file.name} is already registered.")
                continue

            print(f"  [➔] Warping: {target_file.name}...")
            
            try:
                moving_img = ants.image_read(str(target_file))

                # Run the actual warp matrix calculations
                registration = ants.registration(
                    fixed=fixed_img, 
                    moving=moving_img, 
                    type_of_transform='SyN'
                )

                ants.image_write(registration['warpedmovout'], str(output_file))
                print(f"    [✓] Success!")

            except Exception as e:
                print(f"    [FAILED] Could not complete {target_file.name}: {e}")

    print("\n" + "=" * 60)
    print("HPC batch execution complete.")
    print("=" * 60)

if __name__ == "__main__":
    main()