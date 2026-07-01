import pydicom as dicom
import os

def comprehensive_dicom_analysis(dicom_path):
    """Comprehensive analysis of DICOM file for conversion considerations"""
    
    plan = dicom.read_file(dicom_path)
    
    print("=== COMPREHENSIVE DICOM CONVERSION ANALYSIS ===")
    print(f"File: {dicom_path}")
    print()
    
    # 1. BASIC IMAGE CHARACTERISTICS
    print("1. BASIC IMAGE CHARACTERISTICS:")
    print(f"   Modality: {getattr(plan, 'Modality', 'Unknown')}")
    print(f"   Image Type: {getattr(plan, 'ImageType', 'Unknown')}")
    print(f"   Body Part: {getattr(plan, 'BodyPartExamined', 'Unknown')}")
    print(f"   Study Description: {getattr(plan, 'StudyDescription', 'Unknown')}")
    print()
    
    # 2. PIXEL DATA CHARACTERISTICS
    print("2. PIXEL DATA CHARACTERISTICS:")
    print(f"   Rows: {getattr(plan, 'Rows', 'Unknown')}")
    print(f"   Columns: {getattr(plan, 'Columns', 'Unknown')}")
    print(f"   Bits Allocated: {getattr(plan, 'BitsAllocated', 'Unknown')}")
    print(f"   Bits Stored: {getattr(plan, 'BitsStored', 'Unknown')}")
    print(f"   High Bit: {getattr(plan, 'HighBit', 'Unknown')}")
    print(f"   Pixel Representation: {getattr(plan, 'PixelRepresentation', 'Unknown')}")
    print(f"   Samples Per Pixel: {getattr(plan, 'SamplesPerPixel', 'Unknown')}")
    print()
    
    # 3. PHOTOMETRIC AND INTENSITY CHARACTERISTICS
    print("3. PHOTOMETRIC AND INTENSITY CHARACTERISTICS:")
    print(f"   Photometric Interpretation: {getattr(plan, 'PhotometricInterpretation', 'Unknown')}")
    print(f"   Pixel Intensity Relationship: {getattr(plan, 'PixelIntensityRelationship', 'Unknown')}")
    print(f"   Pixel Intensity Relationship Sign: {getattr(plan, 'PixelIntensityRelationshipSign', 'Unknown')}")
    print(f"   Presentation LUT Shape: {getattr(plan, 'PresentationLUTShape', 'Unknown')}")
    print()
    
    # 4. WINDOW/LEVEL SETTINGS
    print("4. WINDOW/LEVEL SETTINGS:")
    print(f"   Window Center: {getattr(plan, 'WindowCenter', 'Not present')}")
    print(f"   Window Width: {getattr(plan, 'WindowWidth', 'Not present')}")
    print()
    
    # 5. RESCALE PARAMETERS
    print("5. RESCALE PARAMETERS:")
    print(f"   Rescale Slope: {getattr(plan, 'RescaleSlope', 'Not present')}")
    print(f"   Rescale Intercept: {getattr(plan, 'RescaleIntercept', 'Not present')}")
    print(f"   Rescale Type: {getattr(plan, 'RescaleType', 'Not present')}")
    print()
    
    # 6. SPATIAL CHARACTERISTICS
    print("6. SPATIAL CHARACTERISTICS:")
    print(f"   Pixel Spacing: {getattr(plan, 'PixelSpacing', 'Not present')}")
    print(f"   Imager Pixel Spacing: {getattr(plan, 'ImagerPixelSpacing', 'Not present')}")
    print(f"   Field of View Shape: {getattr(plan, 'FieldOfViewShape', 'Not present')}")
    print(f"   Field of View Dimensions: {getattr(plan, 'FieldOfViewDimensions', 'Not present')}")
    print()
    
    # 7. ACQUISITION PARAMETERS
    print("7. ACQUISITION PARAMETERS:")
    print(f"   Exposure Index: {getattr(plan, 'ExposureIndex', 'Not present')}")
    print(f"   Target Exposure Index: {getattr(plan, 'TargetExposureIndex', 'Not present')}")
    print(f"   Deviation Index: {getattr(plan, 'DeviationIndex', 'Not present')}")
    print(f"   Sensitivity: {getattr(plan, 'Sensitivity', 'Not present')}")
    print()
    
    # 8. ANNOTATION AND DISPLAY
    print("8. ANNOTATION AND DISPLAY:")
    print(f"   Burned In Annotation: {getattr(plan, 'BurnedInAnnotation', 'Not present')}")
    print(f"   Image Display Format: {getattr(plan, 'ImageDisplayFormat', 'Not present')}")
    print(f"   Film Orientation: {getattr(plan, 'FilmOrientation', 'Not present')}")
    print()
    
    # 9. COMPRESSION
    print("9. COMPRESSION:")
    print(f"   Lossy Image Compression: {getattr(plan, 'LossyImageCompression', 'Not present')}")
    print()
    
    # 10. CONVERSION RECOMMENDATIONS
    print("10. CONVERSION RECOMMENDATIONS:")
    
    # Check for signed vs unsigned
    pixel_rep = getattr(plan, 'PixelRepresentation', None)
    if pixel_rep == 0:
        print("   ✓ Pixel data is UNSIGNED (standard)")
    elif pixel_rep == 1:
        print("   ⚠ Pixel data is SIGNED - may need special handling")
    else:
        print("   ? Pixel representation unknown")
    
    # Check bit depth handling
    bits_stored = getattr(plan, 'BitsStored', None)
    bits_allocated = getattr(plan, 'BitsAllocated', None)
    if bits_stored and bits_allocated:
        print(f"   ✓ Bit depth: {bits_stored} bits stored out of {bits_allocated} allocated")
        if bits_stored < bits_allocated:
            print(f"   ⚠ Note: Only {bits_stored} bits contain meaningful data")
    
    # Check for LOG relationship
    if getattr(plan, 'PixelIntensityRelationship', None) == 'LOG':
        print("   ⚠ LOG intensity relationship detected - use window/level for proper display")
    
    # Check for MONOCHROME1
    if getattr(plan, 'PhotometricInterpretation', None) == 'MONOCHROME1':
        print("   ⚠ MONOCHROME1 detected - image inversion required")
    
    # Check for INVERSE presentation
    if getattr(plan, 'PresentationLUTShape', None) == 'INVERSE':
        print("   ⚠ INVERSE presentation detected - image inversion required")
    
    # Check for compression
    if getattr(plan, 'LossyImageCompression', None) == '01':
        print("   ⚠ Lossy compression detected - image quality may be reduced")
    
    # Check for annotations
    if getattr(plan, 'BurnedInAnnotation', None) == 'YES':
        print("   ⚠ Burned-in annotations detected - consider preserving in output")
    
    print()
    
    # 11. POTENTIAL ISSUES
    print("11. POTENTIAL CONVERSION ISSUES:")
    
    issues = []
    
    # Check for missing critical tags
    if not hasattr(plan, 'WindowCenter') or not hasattr(plan, 'WindowWidth'):
        issues.append("Missing window/level settings - may need fallback normalization")
    
    if not hasattr(plan, 'RescaleSlope') or not hasattr(plan, 'RescaleIntercept'):
        issues.append("Missing rescale parameters - using default values")
    
    # Check for unusual bit depths
    if bits_stored and bits_stored > 16:
        issues.append(f"High bit depth ({bits_stored} bits) - may need special handling")
    
    # Check for unusual image dimensions
    rows = getattr(plan, 'Rows', None)
    cols = getattr(plan, 'Columns', None)
    if rows and cols and (rows > 10000 or cols > 10000):
        issues.append(f"Large image dimensions ({rows}x{cols}) - may cause memory issues")
    
    if issues:
        for issue in issues:
            print(f"   ⚠ {issue}")
    else:
        print("   ✓ No obvious conversion issues detected")
    
    print()
    
    return plan

if __name__ == "__main__":
    dicom_path = "/home/ch215616/ww/code/llm/experiments/yolov7/create_wrist_fracture_dataset/extraction_aug_2025/dcm/26152148/DX.1.2.392.200036.9125.4.0.2535480170.369261468.3253705757"
    
    if os.path.exists(dicom_path):
        plan = comprehensive_dicom_analysis(dicom_path)
    else:
        print(f"DICOM file not found: {dicom_path}") 