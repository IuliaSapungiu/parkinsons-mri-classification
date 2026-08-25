import ants
import os
import argparse

def main():
    # 1. Set up the script to accept terminal commands (-i and -o)
    parser = argparse.ArgumentParser(description="Register MRI to MNI template using ANTs")
    parser.add_argument('-i', '--input', required=True, help="Path to input skull-stripped NIfTI")
    parser.add_argument('-o', '--output', required=True, help="Path to save registered NIfTI")
    args = parser.parse_args()

    input_file = args.input
    output_file = args.output

    # 2. Automatically create the necessary output folders
    output_dir = os.path.dirname(output_file)
    if output_dir:  # Ensures it doesn't crash if running in the current directory
        os.makedirs(output_dir, exist_ok=True)

    print(f"Loading moving image: {input_file}")
    moving_img = ants.image_read(input_file)

    print("Loading standard MNI template...")
    # ants.get_ants_data('mni') automatically fetches a standard template!
    fixed_img = ants.image_read(ants.get_ants_data('mni'))

    print("Running ANTs SyN Registration (this may take 2 to 5 minutes)...")
    # 'SyN' (Symmetric Normalization) is the gold-standard non-linear warp for MRI
    registration = ants.registration(fixed=fixed_img, moving=moving_img, type_of_transform='SyN')

    print(f"Saving registered image to: {output_file}")
    ants.image_write(registration['warpedmovout'], output_file)
    
    print("Success! Registration complete.")

if __name__ == "__main__":
    main()