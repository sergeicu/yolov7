import os
import png
import pydicom as dicom
import numpy as np
from mritopng_v5_optimized import mri_to_png_optimized, convert_file_optimized


def apply_inversion_controlled(plan, pixel_array, force_inversion=False, skip_inversion=False):
    """Apply inversion with control over when to apply it"""
    
    if skip_inversion:
        print("Skipping inversion as requested")
        return pixel_array
    
    if force_inversion:
        pixel_array = np.max(pixel_array) - pixel_array
        print("Applied forced inversion")
        return pixel_array
    
    # Original logic from your function
    inversion_applied = False
    
    # Check photometric interpretation first
    if hasattr(plan, 'PhotometricInterpretation'):
        if plan.PhotometricInterpretation == 'MONOCHROME1':
            pixel_array = np.max(pixel_array) - pixel_array
            inversion_applied = True
            print("Applied MONOCHROME1 inversion")
    
    # Check presentation LUT shape only if inversion hasn't been applied yet
    if not inversion_applied and hasattr(plan, 'PresentationLUTShape'):
        if plan.PresentationLUTShape == 'INVERSE':
            pixel_array = np.max(pixel_array) - pixel_array
            inversion_applied = True
            print("Applied INVERSE presentation")
    
    return pixel_array


def mri_to_png_with_inversion_control(mri_file, png_file, force_inversion=False, skip_inversion=False):
    """Modified version that allows control over inversion"""
    
    # Extracting data from the mri file
    plan = dicom.read_file(mri_file)
    
    # Get pixel data directly as numpy array
    pixel_array = plan.pixel_array.astype(float)
    
    print(f"Original pixel range: {np.min(pixel_array):.2f} to {np.max(pixel_array):.2f}")
    
    # Apply controlled inversion
    pixel_array = apply_inversion_controlled(plan, pixel_array, force_inversion, skip_inversion)
    
    print(f"After inversion: {np.min(pixel_array):.2f} to {np.max(pixel_array):.2f}")
    
    # Min-max normalization
    pixel_min = np.min(pixel_array)
    pixel_max = np.max(pixel_array)
    
    if pixel_max > pixel_min:
        pixel_array = ((pixel_array - pixel_min) / (pixel_max - pixel_min)) * 255
    else:
        pixel_array = np.zeros_like(pixel_array)
    
    # Clip to valid range and convert to uint8
    pixel_array = np.clip(pixel_array, 0, 255).astype(np.uint8)
    
    print(f"After normalization: {np.min(pixel_array)} to {np.max(pixel_array)}")
    
    # Convert to list of lists for PNG writer
    image_2d_scaled = pixel_array.tolist()
    
    # Writing the PNG file
    w = png.Writer(plan.Columns, plan.Rows, greyscale=True)
    w.write(png_file, image_2d_scaled)


def convert_file_with_inversion_control(mri_file_path, png_file_path, force_inversion=False, skip_inversion=False, overwrite=True):
    """Convert with inversion control"""
    
    if not os.path.exists(mri_file_path):
        raise Exception(f'File "{mri_file_path}" does not exist')

    if os.path.exists(png_file_path) and overwrite:
        os.remove(png_file_path)

    mri_file = open(mri_file_path, 'rb')
    png_file = open(png_file_path, 'wb')

    mri_to_png_with_inversion_control(mri_file, png_file, force_inversion, skip_inversion)

    png_file.close()


def main():
    # Path to the DICOM file
    dcm_file = "s20250812_inversion_normalizatoin_experiments/dcm_image"
    
    if not os.path.exists(dcm_file):
        print(f"Error: DICOM file not found at {dcm_file}")
        return
    
    print("=== Converting DICOM image with and without inversion ===\n")
    
    # Convert with default inversion (original behavior)
    print("1. Converting with DEFAULT inversion behavior:")
    convert_file_with_inversion_control(
        dcm_file, 
        "s20250812_inversion_normalizatoin_experiments/result_default_inversion.png"
    )
    print("\n" + "="*50 + "\n")
    
    # Convert with forced inversion
    print("2. Converting with FORCED inversion:")
    convert_file_with_inversion_control(
        dcm_file, 
        "s20250812_inversion_normalizatoin_experiments/result_forced_inversion.png",
        force_inversion=True
    )
    print("\n" + "="*50 + "\n")
    
    # Convert without any inversion
    print("3. Converting with NO inversion:")
    convert_file_with_inversion_control(
        dcm_file, 
        "s20250812_inversion_normalizatoin_experiments/result_no_inversion.png",
        skip_inversion=True
    )
    print("\n" + "="*50 + "\n")
    
    print("Conversion complete! Check the following files:")
    print("- result_default_inversion.png (original behavior)")
    print("- result_forced_inversion.png (always inverted)")
    print("- result_no_inversion.png (never inverted)")


if __name__ == "__main__":
    main()