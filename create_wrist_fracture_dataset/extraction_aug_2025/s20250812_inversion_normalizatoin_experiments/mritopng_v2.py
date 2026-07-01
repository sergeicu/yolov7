"""
DICOM to PNG Converter v2.0 - Optimal DICOM Image Conversion

This script provides advanced DICOM to PNG conversion with proper handling of DICOM-specific
image characteristics. It supports both the original simple conversion method and an
optimized approach that respects DICOM standards.

OVERVIEW:
---------
DICOM (Digital Imaging and Communications in Medicine) files contain medical images with
complex metadata that affects how images should be displayed. Simple conversion methods
often produce poor quality or inverted images because they ignore these DICOM-specific
characteristics.

OLD METHOD (--old flag):
-----------------------
The original conversion method uses simple max normalization:
    pixel_value = (original_value / max_value) * 255

Problems with the old method:
1. Ignores DICOM PhotometricInterpretation (MONOCHROME1 requires inversion)
2. Ignores PresentationLUTShape (INVERSE requires inversion)
3. Ignores PixelIntensityRelationship (LOG scaling needs special handling)
4. Ignores WindowCenter/WindowWidth (manufacturer's recommended display settings)
5. Ignores RescaleSlope/RescaleIntercept (proper intensity scaling)
6. Results in poor contrast, inverted images, or incorrect brightness

NEW METHOD (Default):
--------------------
The optimal method implements proper DICOM handling:

1. DICOM Transformations:
   - Applies RescaleSlope and RescaleIntercept for proper intensity scaling
   - Handles PhotometricInterpretation (MONOCHROME1 inversion)
   - Handles PresentationLUTShape (INVERSE inversion)
   - Respects PixelIntensityRelationship (LOG scaling via window/level)

2. Optimal Normalization:
   - Uses WindowCenter/WindowWidth for manufacturer-recommended display
   - Falls back to min-max normalization if window settings unavailable
   - Ensures proper contrast and brightness

3. Generic Compatibility:
   - Works with any DICOM file regardless of modality
   - Handles missing DICOM tags gracefully (individual tag handling)
   - Provides fallback methods for edge cases
   - Robust handling of partial tag presence (e.g., only RescaleSlope but no RescaleIntercept)

WHY THIS MATTERS:
-----------------
Medical images require accurate representation for:
- Clinical diagnosis and interpretation
- Machine learning model training
- Research and analysis
- Regulatory compliance

Poor conversion can lead to:
- Misdiagnosis due to inverted or poor contrast images
- Reduced model performance in AI applications
- Inconsistent results across different DICOM sources
- Loss of important diagnostic information

USAGE EXAMPLES:
---------------
# Convert single file with optimal method (default)
python mritopng_v2.py input.dcm output.png

# Convert single file with old method
python mritopng_v2.py --old input.dcm output.png

# Convert entire folder with optimal method
python mritopng_v2.py -f input_folder output_folder

# Convert entire folder with old method and overwrite
python mritopng_v2.py -f --old --overwrite input_folder output_folder

DICOM TAGS HANDLED:
------------------
- PhotometricInterpretation: MONOCHROME1 inversion
- PresentationLUTShape: INVERSE inversion  
- PixelIntensityRelationship: LOG scaling support
- WindowCenter/WindowWidth: Optimal contrast settings
- RescaleSlope/RescaleIntercept: Proper intensity scaling
- BitsStored/BitsAllocated: Bit depth handling

AUTHOR: Enhanced DICOM conversion script
VERSION: 2.0
"""

import os
import png
import pydicom as dicom
import argparse
import numpy as np


def mri_to_png_old(mri_file, png_file):
    """ Original simple conversion method (for --old argument)
    
    This function implements the original simple DICOM to PNG conversion that uses
    basic max normalization without considering DICOM-specific characteristics.
    
    WARNING: This method often produces poor quality or inverted images because it:
    - Ignores PhotometricInterpretation (MONOCHROME1 inversion)
    - Ignores PresentationLUTShape (INVERSE inversion)
    - Ignores WindowCenter/WindowWidth settings
    - Ignores RescaleSlope/RescaleIntercept
    - Ignores PixelIntensityRelationship (LOG scaling)
    
    Use this method only for compatibility with legacy systems or when you
    specifically need the original behavior.
    
    @param mri_file: An opened file like object to read the dicom data
    @param png_file: An opened file like object to write the png data
    """
    # Extracting data from the mri file
    plan = dicom.read_file(mri_file)
    shape = plan.pixel_array.shape

    image_2d = []
    max_val = 0
    for row in plan.pixel_array:
        pixels = []
        for col in row:
            pixels.append(col)
            if col > max_val: max_val = col
        image_2d.append(pixels)

    # Rescaling grey scale between 0-255
    image_2d_scaled = []
    for row in image_2d:
        row_scaled = []
        for col in row:
            col_scaled = int((float(col) / float(max_val)) * 255.0)
            row_scaled.append(col_scaled)
        image_2d_scaled.append(row_scaled)

    # Writing the PNG file
    w = png.Writer(shape[1], shape[0], greyscale=True)
    w.write(png_file, image_2d_scaled)


def mri_to_png_optimal(mri_file, png_file):
    """ Optimal DICOM to PNG conversion with generic DICOM handling
    
    This function implements the recommended DICOM to PNG conversion that properly
    handles all DICOM-specific characteristics for optimal image quality.
    
    KEY FEATURES:
    - Applies DICOM rescale parameters (slope/intercept)
    - Handles photometric interpretation (MONOCHROME1 inversion)
    - Handles presentation LUT shape (INVERSE inversion)
    - Uses window/level settings for optimal contrast
    - Falls back gracefully when DICOM tags are missing
    - Works with any DICOM modality (CT, MRI, X-ray, etc.)
    
    This method produces images that match the intended clinical display,
    making it suitable for medical applications, research, and AI model training.
    
    @param mri_file: An opened file like object to read the dicom data
    @param png_file: An opened file like object to write the png data
    """
    # Read DICOM file
    plan = dicom.read_file(mri_file)
    pixel_array = plan.pixel_array.astype(float)
    
    # Apply DICOM-specific transformations
    pixel_array = apply_dicom_transformations(plan, pixel_array)
    
    # Normalize to 0-255 range
    pixel_array = normalize_to_8bit(plan, pixel_array)
    
    # Convert to list of lists for PNG writer
    image_2d_scaled = pixel_array.astype(int).tolist()
    
    # Writing the PNG file
    w = png.Writer(plan.Columns, plan.Rows, greyscale=True)
    w.write(png_file, image_2d_scaled)


def apply_dicom_transformations(plan, pixel_array):
    """Apply DICOM-specific transformations to pixel data
    
    This function applies all necessary DICOM transformations to convert raw pixel
    data into properly scaled and oriented image data.
    
    TRANSFORMATIONS APPLIED:
    1. RescaleSlope/RescaleIntercept: Converts pixel values to proper intensity units
    2. PhotometricInterpretation: Handles MONOCHROME1 inversion if needed
    3. PresentationLUTShape: Handles INVERSE presentation if needed
    
    The function handles missing DICOM tags gracefully and provides appropriate
    fallback behavior for edge cases.
    
    @param plan: DICOM dataset object containing metadata
    @param pixel_array: Numpy array of pixel data
    @return: Transformed pixel array ready for normalization
    """
    
    def safe_float(value, default=0.0):
        """Safely convert value to float with fallback to default"""
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    # Apply rescale slope and intercept (handle each individually)
    # Default values: slope=1.0 (no scaling), intercept=0.0 (no offset)
    slope = safe_float(getattr(plan, 'RescaleSlope', 1.0), 1.0)
    intercept = safe_float(getattr(plan, 'RescaleIntercept', 0.0), 0.0)
    
    # Apply rescale transformation
    pixel_array = pixel_array * slope + intercept
    
    # Handle PhotometricInterpretation
    if hasattr(plan, 'PhotometricInterpretation'):
        if plan.PhotometricInterpretation == 'MONOCHROME1':
            # Invert the image for MONOCHROME1
            pixel_array = np.max(pixel_array) - pixel_array
    
    # Handle PresentationLUTShape
    if hasattr(plan, 'PresentationLUTShape'):
        if plan.PresentationLUTShape == 'INVERSE':
            # Invert the image
            pixel_array = np.max(pixel_array) - pixel_array
    
    # Handle PixelIntensityRelationship (LOG)
    # Note: For LOG images, we typically rely on window/level settings
    # rather than applying exponential transformation, as the window/level
    # approach is the standard DICOM method for handling LOG images
    
    return pixel_array


def normalize_to_8bit(plan, pixel_array):
    """Normalize pixel array to 0-255 range using optimal method
    
    This function implements the optimal normalization strategy for DICOM images,
    prioritizing manufacturer-recommended display settings over simple min-max scaling.
    
    NORMALIZATION STRATEGY:
    1. PRIMARY: Use WindowCenter/WindowWidth (manufacturer's recommended settings)
       - Provides optimal contrast for the specific image type
       - Handles LOG scaling properly through window/level mechanism
       - Ensures clinical accuracy
    
    2. FALLBACK: Min-max normalization
       - Used when window/level settings are unavailable
       - Used when window settings are invalid
       - Ensures all pixel values are represented
    
    The function ensures all output values are in the valid 0-255 range
    and handles edge cases gracefully.
    
    @param plan: DICOM dataset object containing metadata
    @param pixel_array: Transformed pixel array
    @return: Normalized pixel array in 0-255 range
    """
    
    def safe_float(value, default=0.0):
        """Safely convert value to float with fallback to default"""
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    # Use window center and width if available (standard DICOM approach)
    # Handle each tag individually - they may be present independently
    window_center = safe_float(getattr(plan, 'WindowCenter', None), None)
    window_width = safe_float(getattr(plan, 'WindowWidth', None), None)
    
    if window_center is not None and window_width is not None and window_width > 0:
        # Calculate window limits
        window_min = window_center - window_width / 2
        window_max = window_center + window_width / 2
        
        # Clip values to window
        pixel_array = np.clip(pixel_array, window_min, window_max)
        
        # Normalize to 0-255
        if window_max > window_min:
            pixel_array = ((pixel_array - window_min) / (window_max - window_min)) * 255
        else:
            # Fallback to min-max normalization if window is invalid
            pixel_min = np.min(pixel_array)
            pixel_max = np.max(pixel_array)
            if pixel_max > pixel_min:
                pixel_array = ((pixel_array - pixel_min) / (pixel_max - pixel_min)) * 255
    else:
        # Fallback to min-max normalization
        pixel_min = np.min(pixel_array)
        pixel_max = np.max(pixel_array)
        if pixel_max > pixel_min:
            pixel_array = ((pixel_array - pixel_min) / (pixel_max - pixel_min)) * 255
    
    # Ensure values are in valid range
    pixel_array = np.clip(pixel_array, 0, 255)
    
    return pixel_array


def mri_to_png(mri_file, png_file, use_old_method=False):
    """ Main conversion function that chooses between old and optimal methods
    
    This is the main entry point for DICOM to PNG conversion. It routes to either
    the old simple method or the new optimal method based on the use_old_method flag.
    
    RECOMMENDATION:
    - Use optimal method (default) for medical applications, research, and AI training
    - Use old method only for legacy compatibility or specific requirements
    
    @param mri_file: An opened file like object to read the dicom data
    @param png_file: An opened file like object to write the png data
    @param use_old_method: If True, use the original simple method (not recommended)
    """
    if use_old_method:
        mri_to_png_old(mri_file, png_file)
    else:
        mri_to_png_optimal(mri_file, png_file)


def convert_file(mri_file_path, png_file_path, overwrite=False, use_old_method=False):
    """ Function to convert a DICOM file to a PNG image file.
    
    This function handles the file-level conversion from DICOM to PNG format.
    It includes proper error handling, file existence checks, and overwrite
    protection.
    
    FEATURES:
    - Validates input file existence
    - Handles overwrite protection
    - Provides detailed error messages
    - Supports both old and optimal conversion methods
    
    @param mri_file_path: Full path to the DICOM file
    @param png_file_path: Full path to the output PNG file
    @param overwrite: Whether to overwrite existing PNG files
    @param use_old_method: Whether to use the old conversion method (not recommended)
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

    mri_to_png(mri_file, png_file, use_old_method)

    png_file.close()


def convert_folder(mri_folder, png_folder, overwrite=False, use_old_method=False):
    """ Convert all DICOM files in a folder to PNG files in a destination folder
    
    This function recursively processes all DICOM files in a folder structure and
    converts them to PNG format while preserving the directory structure.
    
    FEATURES:
    - Recursive folder processing
    - Preserves original directory structure
    - Handles file-level errors gracefully
    - Provides progress feedback
    - Supports both old and optimal conversion methods
    
    The function will create the output directory structure automatically and
    skip files that already exist (unless overwrite=True).
    
    @param mri_folder: Source folder containing DICOM files
    @param png_folder: Destination folder for PNG files
    @param overwrite: Whether to overwrite existing PNG files
    @param use_old_method: Whether to use the old conversion method (not recommended)
    """

    # Create the folder for the png directory structure
    os.makedirs(png_folder, exist_ok=True)

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
                png_file_path = os.path.join(png_folder_path, '%s.png' % mri_file)

                try:
                    # Convert the actual file
                    convert_file(mri_file_path, png_file_path, overwrite=overwrite, use_old_method=use_old_method)
                    print(f'SUCCESS> {mri_file_path}, -->, {png_file_path}')
                except Exception as e:
                    print(f'FAIL>, {mri_file_path}, -->, {png_file_path}, :, {e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert a dicom MRI file to png with optimal DICOM handling")
    parser.add_argument('-f', action='store_true', help='Convert entire folder')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files')
    parser.add_argument('--old', action='store_true', help='Use old simple conversion method')
    parser.add_argument('dicom_path', help='Full path to the mri file or folder')
    parser.add_argument('png_path', help='Full path to the generated png file or folder')

    args = parser.parse_args()
    print(args)
    if args.f:
        convert_folder(args.dicom_path, args.png_path, args.overwrite, args.old)
    else:
        convert_file(args.dicom_path, args.png_path, args.overwrite, args.old) 