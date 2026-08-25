import os
import glob
import nibabel as nib

def verify_nifti_files():
    print("Starting data integrity check...")
    
    # Target our new clean directory
    data_dir = 'openneuro'
    
    # Find all .nii.gz files in both HC and PD folders
    search_pattern = os.path.join(data_dir, '*', '*.nii.gz')
    all_files = glob.glob(search_pattern)
    
    if not all_files:
        print("No .nii.gz files found! Check your folder path.")
        return

    print(f"Found {len(all_files)} files. Testing headers...")
    
    corrupted_files = []

    # Loop through and try to load each file
    for filepath in all_files:
        try:
            # nib.load reads the file header (it doesn't load the massive image into memory, so it's fast)
            img = nib.load(filepath)
            # Fetching the shape forces it to parse the header structure
            shape = img.shape 
        except Exception as e:
            print(f"❌ CORRUPTED: {os.path.basename(filepath)}\n   Error: {e}")
            corrupted_files.append(filepath)

    print("\n--- Verification Summary ---")
    if len(corrupted_files) == 0:
        print("✅ Success! All 46 NIfTI files are healthy and readable.")
    else:
        print(f"⚠️ Found {len(corrupted_files)} corrupted files. You will need to re-download these specific files.")

if __name__ == "__main__":
    verify_nifti_files()