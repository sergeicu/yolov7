import os
import png
import pydicom as dicom
import argparse
import numpy as np


def apply_inversion_if_needed(plan, pixel_array):
    """Apply inversion only once if either PresentationLUTShape or PhotometricInterpretation indicates it"""
    
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


def minmax_normalization(pixel_array):
    """Apply min-max normalization"""
    pixel_min = np.min(pixel_array)
    pixel_max = np.max(pixel_array)
    
    if pixel_max > pixel_min:
        pixel_array = ((pixel_array - pixel_min) / (pixel_max - pixel_min)) * 255
    else:
        pixel_array = np.zeros_like(pixel_array)
    
    return np.clip(pixel_array, 0, 255).astype(np.uint8)


def mri_to_png_v4(mri_file, output_path):
    """Enhanced function to convert from a DICOM image to PNG with min-max normalization"""
    
    # Extract data from the mri file
    plan = dicom.read_file(mri_file)
    
    # Validate DICOM file
    if not hasattr(plan, 'pixel_array'):
        raise ValueError("DICOM file does not contain pixel data")
    
    if not hasattr(plan, 'Rows') or not hasattr(plan, 'Columns'):
        raise ValueError("DICOM file missing image dimensions")
    
    pixel_array = plan.pixel_array.astype(float)
    
    print(f"Original pixel range: {np.min(pixel_array):.2f} to {np.max(pixel_array):.2f}")
    
    # Step 1: Apply rescale parameters (HU values for CT, etc.)
    slope = float(getattr(plan, 'RescaleSlope', 1.0))
    intercept = float(getattr(plan, 'RescaleIntercept', 0.0))
    if slope != 1.0 or intercept != 0.0:
        pixel_array = pixel_array * slope + intercept
        print(f"Applied rescale: slope={slope}, intercept={intercept}")
        print(f"After rescale: {np.min(pixel_array):.2f} to {np.max(pixel_array):.2f}")
    
    # Step 2: Handle pixel representation (signed/unsigned)
    pixel_rep = getattr(plan, 'PixelRepresentation', 0)
    if pixel_rep == 1:  # Signed
        # Convert to appropriate signed type if needed
        if pixel_array.dtype != np.int16:
            pixel_array = pixel_array.astype(np.int16)
        print("Handled signed pixel representation")
    
    # Step 3: Handle bit depth if needed
    bits_stored = getattr(plan, 'BitsStored', None)
    bits_allocated = getattr(plan, 'BitsAllocated', None)
    if bits_stored and bits_allocated and bits_stored < bits_allocated:
        shift = bits_allocated - bits_stored
        # Convert to integer for bit shifting, then back to float
        pixel_array = pixel_array.astype(np.int32) >> shift
        pixel_array = pixel_array.astype(float)
        max_value = (1 << bits_stored) - 1
        pixel_array = np.clip(pixel_array, 0, max_value)
        print(f"Applied bit depth handling: stored={bits_stored}, allocated={bits_allocated}")
    
    # Step 4: Apply inversion if needed (only once)
    pixel_array = apply_inversion_if_needed(plan, pixel_array)
    
    print(f"After all transformations: {np.min(pixel_array):.2f} to {np.max(pixel_array):.2f}")
    
    # Step 5: Apply min-max normalization
    print("Applying min-max normalization...")
    pixel_array_normalized = minmax_normalization(pixel_array)
    
    # Step 6: Save PNG file
    with open(output_path, 'wb') as png_file:
        w = png.Writer(plan.Columns, plan.Rows, greyscale=True)
        w.write(png_file, pixel_array_normalized.tolist())
    print(f"Saved PNG: {output_path}")
    
    return output_path


def convert_file_v4(mri_file_path, output_path, overwrite=False):
    """Convert an MRI binary file to PNG image with min-max normalization"""
    
    # Making sure that the mri file exists
    if not os.path.exists(mri_file_path):
        raise Exception(f'File "{mri_file_path}" does not exist')
    
    # Check if output file already exists
    if not overwrite and os.path.exists(output_path):
        print(f"Skipping {mri_file_path} - output file already exists")
        return None
    
    # Delete existing file if overwrite is True
    if overwrite and os.path.exists(output_path):
        os.remove(output_path)
    
    try:
        with open(mri_file_path, 'rb') as mri_file:
            output_path = mri_to_png_v4(mri_file, output_path)
        return output_path
    except Exception as e:
        print(f"Error processing {mri_file_path}: {e}")
        raise


def convert_folder_v4(mri_folder, output_folder, overwrite=False):
    """Convert all MRI files in a folder to PNG files with min-max normalization"""
    
    # Create the output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Recursively traverse all sub-folders in the path
    for mri_sub_folder, subdirs, files in os.walk(mri_folder):
        for mri_file in os.listdir(mri_sub_folder):
            mri_file_path = os.path.join(mri_sub_folder, mri_file)
            
            # Make sure path is an actual file
            if os.path.isfile(mri_file_path):
                # Create output path
                output_filename = f"{mri_file}.png"
                output_path = os.path.join(output_folder, output_filename)
                
                try:
                    # Convert the actual file
                    output_path = convert_file_v4(mri_file_path, output_path, overwrite=overwrite)
                    if output_path:
                        print(f'SUCCESS> {mri_file_path} --> {output_path}')
                except Exception as e:
                    print(f'FAIL> {mri_file_path}: {e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert DICOM MRI files to PNG with min-max normalization")
    parser.add_argument('-f', action='store_true', help='Convert entire folder')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files')
    parser.add_argument('dicom_path', help='Full path to the DICOM file or folder')
    parser.add_argument('output_path', help='Full path to the output PNG file or folder')
    
    args = parser.parse_args()
    print(args)
    
    if args.f:
        # Folder conversion
        convert_folder_v4(args.dicom_path, args.output_path, args.overwrite)
    else:
        # Single file conversion
        try:
            output_path = convert_file_v4(args.dicom_path, args.output_path, args.overwrite)
            if output_path:
                print(f"Conversion complete!")
                print(f"Output: {output_path}")
        except Exception as e:
            print(f"Error: {e}") 