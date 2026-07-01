import os
import png
import pydicom as dicom
import numpy as np


def test_different_log_approaches(dicom_path):
    """Test different approaches to handle LOG scaling"""
    
    plan = dicom.read_file(dicom_path)
    pixel_array = plan.pixel_array.astype(float)
    
    print(f"Original pixel range: {np.min(pixel_array)} to {np.max(pixel_array)}")
    
    # Approach 1: No LOG handling (original)
    pixel_array_1 = pixel_array.copy()
    if hasattr(plan, 'PhotometricInterpretation') and plan.PhotometricInterpretation == 'MONOCHROME1':
        pixel_array_1 = np.max(pixel_array_1) - pixel_array_1
    if hasattr(plan, 'PresentationLUTShape') and plan.PresentationLUTShape == 'INVERSE':
        pixel_array_1 = np.max(pixel_array_1) - pixel_array_1
    
    # Approach 2: Apply exponential transformation for LOG
    pixel_array_2 = pixel_array.copy()
    if hasattr(plan, 'PixelIntensityRelationship') and plan.PixelIntensityRelationship == 'LOG':
        # Apply exponential transformation
        pixel_array_2 = np.exp(pixel_array_2 / 1000.0)  # Scale factor to prevent overflow
        print(f"After exp transformation: {np.min(pixel_array_2)} to {np.max(pixel_array_2)}")
    
    if hasattr(plan, 'PhotometricInterpretation') and plan.PhotometricInterpretation == 'MONOCHROME1':
        pixel_array_2 = np.max(pixel_array_2) - pixel_array_2
    if hasattr(plan, 'PresentationLUTShape') and plan.PresentationLUTShape == 'INVERSE':
        pixel_array_2 = np.max(pixel_array_2) - pixel_array_2
    
    # Approach 3: Use window/level normalization (standard DICOM approach)
    pixel_array_3 = pixel_array.copy()
    if hasattr(plan, 'WindowCenter') and hasattr(plan, 'WindowWidth'):
        window_center = float(plan.WindowCenter)
        window_width = float(plan.WindowWidth)
        window_min = window_center - window_width / 2
        window_max = window_center + window_width / 2
        
        # Clip to window
        pixel_array_3 = np.clip(pixel_array_3, window_min, window_max)
        # Normalize to 0-255
        pixel_array_3 = ((pixel_array_3 - window_min) / (window_max - window_min)) * 255
    else:
        # Fallback to min-max normalization
        pixel_min = np.min(pixel_array_3)
        pixel_max = np.max(pixel_array_3)
        pixel_array_3 = ((pixel_array_3 - pixel_min) / (pixel_max - pixel_min)) * 255
    
    # Apply inversions
    if hasattr(plan, 'PhotometricInterpretation') and plan.PhotometricInterpretation == 'MONOCHROME1':
        pixel_array_3 = np.max(pixel_array_3) - pixel_array_3
    if hasattr(plan, 'PresentationLUTShape') and plan.PresentationLUTShape == 'INVERSE':
        pixel_array_3 = np.max(pixel_array_3) - pixel_array_3
    
    # Approach 4: Simple min-max normalization (your original approach)
    pixel_array_4 = pixel_array.copy()
    pixel_min = np.min(pixel_array_4)
    pixel_max = np.max(pixel_array_4)
    pixel_array_4 = ((pixel_array_4 - pixel_min) / (pixel_max - pixel_min)) * 255
    
    # Apply inversions
    if hasattr(plan, 'PhotometricInterpretation') and plan.PhotometricInterpretation == 'MONOCHROME1':
        pixel_array_4 = np.max(pixel_array_4) - pixel_array_4
    if hasattr(plan, 'PresentationLUTShape') and plan.PresentationLUTShape == 'INVERSE':
        pixel_array_4 = np.max(pixel_array_4) - pixel_array_4
    
    # Save all approaches
    approaches = [
        ("approach1_no_log.png", pixel_array_1),
        ("approach2_with_exp.png", pixel_array_2),
        ("approach3_window_level.png", pixel_array_3),
        ("approach4_simple_minmax.png", pixel_array_4)
    ]
    
    for filename, data in approaches:
        # Normalize to 0-255 if not already done
        if np.max(data) > 255:
            data_min = np.min(data)
            data_max = np.max(data)
            data = ((data - data_min) / (data_max - data_min)) * 255
        
        # Clip to valid range
        data = np.clip(data, 0, 255)
        
        # Convert to list and save
        image_2d = data.astype(int).tolist()
        with open(filename, 'wb') as png_file:
            w = png.Writer(plan.Columns, plan.Rows, greyscale=True)
            w.write(png_file, image_2d)
        
        print(f"Created {filename}")


if __name__ == "__main__":
    dicom_path = "/home/ch215616/ww/code/llm/experiments/yolov7/create_wrist_fracture_dataset/extraction_aug_2025/dcm/26152148/DX.1.2.392.200036.9125.4.0.2535480170.369261468.3253705757"
    test_different_log_approaches(dicom_path) 