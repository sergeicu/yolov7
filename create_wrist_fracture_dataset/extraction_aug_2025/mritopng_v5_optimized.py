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


def mri_to_png_optimized(mri_file, png_file):
    """ Optimized function to convert from a DICOM image to png using pure numpy operations

        @param mri_file: An opened file like object to read the dicom data
        @param png_file: An opened file like object to write the png data
    """

    # Extracting data from the mri file
    plan = dicom.read_file(mri_file)
    
    # Get pixel data directly as numpy array (much faster!)
    pixel_array = plan.pixel_array.astype(float)
    
    print(f"Original pixel range: {np.min(pixel_array):.2f} to {np.max(pixel_array):.2f}")
    
    # Apply inversion fix (default behavior) - pure numpy operations
    pixel_array = apply_inversion_if_needed(plan, pixel_array)
    
    print(f"After inversion: {np.min(pixel_array):.2f} to {np.max(pixel_array):.2f}")
    
    # Min-max normalization using pure numpy (much faster!)
    pixel_min = np.min(pixel_array)
    pixel_max = np.max(pixel_array)
    
    if pixel_max > pixel_min:
        # Vectorized normalization - much faster than loops!
        pixel_array = ((pixel_array - pixel_min) / (pixel_max - pixel_min)) * 255
    else:
        # Handle case where all pixels are the same
        pixel_array = np.zeros_like(pixel_array)
    
    # Clip to valid range and convert to uint8
    pixel_array = np.clip(pixel_array, 0, 255).astype(np.uint8)
    
    print(f"After normalization: {np.min(pixel_array)} to {np.max(pixel_array)}")
    
    # Convert to list of lists for PNG writer (required by png library)
    image_2d_scaled = pixel_array.tolist()
    
    # Writing the PNG file
    w = png.Writer(plan.Columns, plan.Rows, greyscale=True)
    w.write(png_file, image_2d_scaled)


def convert_file_optimized(mri_file_path, png_file_path, overwrite=False):
    """ Function to convert an MRI binary file to a
        PNG image file.

        @param mri_file_path: Full path to the mri file
        @param png_file_path: Full path to the png file
        @param overwrite: Whether to overwrite existing files
    """

    # Making sure that the mri file exists
    if not os.path.exists(mri_file_path):
        raise Exception('File "%s" does not exists' % mri_file_path)

    # Skip existing files 
    if os.path.exists(png_file_path) and not overwrite:
        return 
        #raise Exception('File "%s" already exists' % png_file_path)
    # or delete and re-run the conversion
    elif os.path.exists(png_file_path):
        os.remove(png_file_path)

    mri_file = open(mri_file_path, 'rb')
    png_file = open(png_file_path, 'wb')

    mri_to_png_optimized(mri_file, png_file)

    png_file.close()


def convert_folder_optimized(mri_folder, png_folder, overwrite=False):
    """ Convert all MRI files in a folder to png files
        in a destination folder
    """

    # Create the folder for the png directory structure
    os.makedirs(png_folder)

    # Recursively traverse all sub-folders in the path
    for mri_sub_folder, subdirs, files in os.walk(mri_folder):
        for mri_file in os.listdir(mri_sub_folder):
            mri_file_path = os.path.join(mri_sub_folder, mri_file)

            # Make sure path is an actual file
            if os.path.isfile(mri_file_path):

                # Replicate the original file structure
                rel_path = os.path.relpath(mri_sub_folder, mri_folder)
                png_folder_path = os.path.join(png_folder, rel_path)
                if not os.path.exists(png_folder_path):
                    os.makedirs(png_folder_path)
                
                # Add 'fixed' subfolder (default behavior)
                png_folder_path = os.path.join(png_folder_path, 'fixed')
                if not os.path.exists(png_folder_path):
                    os.makedirs(png_folder_path)
                
                png_file_path = os.path.join(png_folder_path, '%s.png' % mri_file)

                try:
                    # Convert the actual file
                    convert_file_optimized(mri_file_path, png_file_path, overwrite=overwrite)
                    print(f'SUCCESS> {mri_file_path}, -->, {png_file_path}')
                except Exception as e:
                    print(f'FAIL>, {mri_file_path}, -->, {png_file_path}, :, {e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert a dicom MRI file to png (optimized version)")
    parser.add_argument('-f', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('dicom_path', help='Full path to the mri file')
    parser.add_argument('png_path', help='Full path to the generated png file')

    args = parser.parse_args()
    print(args)
    if args.f:
        convert_folder_optimized(args.dicom_path, args.png_path, args.overwrite)
    else:
        convert_file_optimized(args.dicom_path, args.png_path, args.overwrite) 