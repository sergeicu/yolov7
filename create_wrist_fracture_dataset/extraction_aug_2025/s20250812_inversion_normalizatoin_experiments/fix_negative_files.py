import os
import png
import pydicom as dicom
import numpy as np
import argparse


def analyze_dicom_characteristics(dicom_file):
    """Analyze DICOM file characteristics to understand display requirements"""
    
    plan = dicom.read_file(dicom_file)
    
    print("=== DICOM Image Analysis ===")
    print(f"PhotometricInterpretation: {getattr(plan, 'PhotometricInterpretation', 'Not found')}")
    print(f"PixelIntensityRelationship: {getattr(plan, 'PixelIntensityRelationship', 'Not found')}")
    print(f"PixelIntensityRelationshipSign: {getattr(plan, 'PixelIntensityRelationshipSign', 'Not found')}")
    print(f"PresentationLUTShape: {getattr(plan, 'PresentationLUTShape', 'Not found')}")
    print(f"WindowCenter: {getattr(plan, 'WindowCenter', 'Not found')}")
    print(f"WindowWidth: {getattr(plan, 'WindowWidth', 'Not found')}")
    print(f"RescaleSlope: {getattr(plan, 'RescaleSlope', 'Not found')}")
    print(f"RescaleIntercept: {getattr(plan, 'RescaleIntercept', 'Not found')}")
    print(f"BitsStored: {getattr(plan, 'BitsStored', 'Not found')}")
    print(f"BitsAllocated: {getattr(plan, 'BitsAllocated', 'Not found')}")
    print(f"HighBit: {getattr(plan, 'HighBit', 'Not found')}")
    print(f"PixelRepresentation: {getattr(plan, 'PixelRepresentation', 'Not found')}")
    
    # Analyze pixel data statistics
    pixel_array = plan.pixel_array
    print(f"\n=== Pixel Data Statistics ===")
    print(f"Pixel data shape: {pixel_array.shape}")
    print(f"Min value: {np.min(pixel_array)}")
    print(f"Max value: {np.max(pixel_array)}")
    print(f"Mean value: {np.mean(pixel_array):.2f}")
    print(f"Standard deviation: {np.std(pixel_array):.2f}")
    
    # Check for LOG relationship implications
    if hasattr(plan, 'PixelIntensityRelationship') and plan.PixelIntensityRelationship == 'LOG':
        print(f"\n=== LOG Relationship Analysis ===")
        print("LOG relationship indicates the pixel values represent logarithmic intensity.")
        print("This means the relationship between pixel values and actual intensity is:")
        print("  intensity = base^(pixel_value / scale_factor)")
        print("For display purposes, we typically want linear intensity, so we might need to:")
        print("  1. Apply exponential transformation: pixel_value = exp(pixel_value)")
        print("  2. Or use appropriate window/level settings")
        print("  3. Or rely on the DICOM viewer's built-in LOG handling")
    
    return plan


def mri_to_png_simple(mri_file, png_file):
    """Original simple conversion method"""
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


def mri_to_png_with_log_handling(mri_file, png_file):
    """Conversion with LOG relationship handling"""
    plan = dicom.read_file(mri_file)
    pixel_array = plan.pixel_array.astype(float)
    
    print(f"Original pixel range: {np.min(pixel_array)} to {np.max(pixel_array)}")
    
    # Handle LOG relationship
    if hasattr(plan, 'PixelIntensityRelationship') and plan.PixelIntensityRelationship == 'LOG':
        print("Applying LOG relationship handling...")
        # For LOG images, we might need to apply exponential transformation
        # But first, let's try without it and see the result
        pixel_array_log = pixel_array.copy()
        
        # Alternative: apply exponential transformation
        # pixel_array = np.exp(pixel_array)
        # print(f"After exp transformation: {np.min(pixel_array)} to {np.max(pixel_array)}")
    
    # Handle MONOCHROME1 inversion
    if hasattr(plan, 'PhotometricInterpretation') and plan.PhotometricInterpretation == 'MONOCHROME1':
        print("Applying MONOCHROME1 inversion...")
        pixel_array = np.max(pixel_array) - pixel_array
    
    # Handle INVERSE presentation
    elif hasattr(plan, 'PresentationLUTShape') and plan.PresentationLUTShape == 'INVERSE':
        print("Applying INVERSE presentation...")
        pixel_array = np.max(pixel_array) - pixel_array
    
    # Normalize using window/level if available
    if hasattr(plan, 'WindowCenter') and hasattr(plan, 'WindowWidth'):
        window_center = float(plan.WindowCenter)
        window_width = float(plan.WindowWidth)
        window_min = window_center - window_width / 2
        window_max = window_center + window_width / 2
        
        print(f"Using window: center={window_center}, width={window_width}")
        print(f"Window range: {window_min} to {window_max}")
        
        # Clip to window
        pixel_array = np.clip(pixel_array, window_min, window_max)
        
        # Normalize to 0-255
        pixel_array = ((pixel_array - window_min) / (window_max - window_min)) * 255
    else:
        # Fallback to min-max normalization
        pixel_min = np.min(pixel_array)
        pixel_max = np.max(pixel_array)
        pixel_array = ((pixel_array - pixel_min) / (pixel_max - pixel_min)) * 255
    
    # Convert to list and write
    image_2d_scaled = pixel_array.astype(int).tolist()
    w = png.Writer(plan.Columns, plan.Rows, greyscale=True)
    w.write(png_file, image_2d_scaled)


def test_normalization_methods(dicom_file, output_prefix="test"):
    """
    Test different normalization methods and save results for comparison.
    
    Args:
        dicom_file: Path to DICOM file
        output_prefix: Prefix for output files
    """
    plan = dicom.read_file(dicom_file)
    pixel_array = plan.pixel_array.astype(float)
    
    print("=== Testing Normalization Methods ===")
    print(f"Original pixel range: {np.min(pixel_array):.2f} to {np.max(pixel_array):.2f}")
    
    # Apply any necessary transformations first (inversion, etc.)
    # Handle MONOCHROME1 inversion
    if hasattr(plan, 'PhotometricInterpretation') and plan.PhotometricInterpretation == 'MONOCHROME1':
        print("Applying MONOCHROME1 inversion...")
        pixel_array = np.max(pixel_array) - pixel_array
    
    # Handle INVERSE presentation (only if MONOCHROME1 not applied)
    elif hasattr(plan, 'PresentationLUTShape') and plan.PresentationLUTShape == 'INVERSE':
        print("Applying INVERSE presentation...")
        pixel_array = np.max(pixel_array) - pixel_array
    
    print(f"After transformations: {np.min(pixel_array):.2f} to {np.max(pixel_array):.2f}")
    
    # Method 1: Window/Level normalization
    if hasattr(plan, 'WindowCenter') and hasattr(plan, 'WindowWidth'):
        window_center = float(plan.WindowCenter)
        window_width = float(plan.WindowWidth)
        window_min = window_center - window_width / 2
        window_max = window_center + window_width / 2
        
        print(f"\n=== Window/Level Method ===")
        print(f"Window Center: {window_center}")
        print(f"Window Width: {window_width}")
        print(f"Window Range: {window_min:.2f} to {window_max:.2f}")
        
        # Clip to window
        pixel_array_window = np.clip(pixel_array, window_min, window_max)
        
        # Normalize to 0-255
        if window_max > window_min:
            pixel_array_window = ((pixel_array_window - window_min) / (window_max - window_min)) * 255
        else:
            print("Warning: Invalid window (max <= min), using min-max fallback")
            pixel_min = np.min(pixel_array_window)
            pixel_max = np.max(pixel_array_window)
            if pixel_max > pixel_min:
                pixel_array_window = ((pixel_array_window - pixel_min) / (pixel_max - pixel_min)) * 255
        
        pixel_array_window = np.clip(pixel_array_window, 0, 255)
        print(f"Final range: {np.min(pixel_array_window):.2f} to {np.max(pixel_array_window):.2f}")
        
        # Save window/level version
        image_2d_window = pixel_array_window.astype(int).tolist()
        with open(f"{output_prefix}_window_level.png", 'wb') as png_file:
            w = png.Writer(plan.Columns, plan.Rows, greyscale=True)
            w.write(png_file, image_2d_window)
        print(f"Saved: {output_prefix}_window_level.png")
    
    # Method 2: Min-Max normalization
    print(f"\n=== Min-Max Method ===")
    pixel_min = np.min(pixel_array)
    pixel_max = np.max(pixel_array)
    print(f"Data range: {pixel_min:.2f} to {pixel_max:.2f}")
    
    if pixel_max > pixel_min:
        pixel_array_minmax = ((pixel_array - pixel_min) / (pixel_max - pixel_min)) * 255
    else:
        print("Warning: All pixel values are the same!")
        pixel_array_minmax = np.zeros_like(pixel_array)
    
    pixel_array_minmax = np.clip(pixel_array_minmax, 0, 255)
    print(f"Final range: {np.min(pixel_array_minmax):.2f} to {np.max(pixel_array_minmax):.2f}")
    
    # Save min-max version
    image_2d_minmax = pixel_array_minmax.astype(int).tolist()
    with open(f"{output_prefix}_min_max.png", 'wb') as png_file:
        w = png.Writer(plan.Columns, plan.Rows, greyscale=True)
        w.write(png_file, image_2d_minmax)
    print(f"Saved: {output_prefix}_min_max.png")
    
    # Method 3: Percentile-based normalization (robust to outliers)
    print(f"\n=== Percentile Method ===")
    p1 = np.percentile(pixel_array, 1)  # 1st percentile
    p99 = np.percentile(pixel_array, 99)  # 99th percentile
    print(f"1st percentile: {p1:.2f}")
    print(f"99th percentile: {p99:.2f}")
    
    if p99 > p1:
        pixel_array_percentile = ((pixel_array - p1) / (p99 - p1)) * 255
        pixel_array_percentile = np.clip(pixel_array_percentile, 0, 255)
    else:
        pixel_array_percentile = np.zeros_like(pixel_array)
    
    print(f"Final range: {np.min(pixel_array_percentile):.2f} to {np.max(pixel_array_percentile):.2f}")
    
    # Save percentile version
    image_2d_percentile = pixel_array_percentile.astype(int).tolist()
    with open(f"{output_prefix}_percentile.png", 'wb') as png_file:
        w = png.Writer(plan.Columns, plan.Rows, greyscale=True)
        w.write(png_file, image_2d_percentile)
    print(f"Saved: {output_prefix}_percentile.png")
    
    return {
        'window_level': pixel_array_window if hasattr(plan, 'WindowCenter') else None,
        'min_max': pixel_array_minmax,
        'percentile': pixel_array_percentile
    }

# Usage example:
if __name__ == "__main__":
    dicom_path = 'dcm_image'  # Your DICOM file path
    
    print("=== Testing Normalization Methods ===")
    results = test_normalization_methods(dicom_path, "normalization_test")
    
    print("\n=== Summary ===")
    print("Generated files:")
    print("- normalization_test_window_level.png (if window/level available)")
    print("- normalization_test_min_max.png")
    print("- normalization_test_percentile.png")
    print("\nCompare these images to see which normalization method works best for your data!")