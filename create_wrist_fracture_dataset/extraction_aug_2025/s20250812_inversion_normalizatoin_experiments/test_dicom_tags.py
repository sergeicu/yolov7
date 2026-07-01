import pydicom as dicom
import os

def analyze_dicom_tags(dicom_path):
    """Analyze DICOM tags to understand tag presence patterns"""
    
    plan = dicom.read_file(dicom_path)
    
    print("=== DICOM Tag Analysis ===")
    print(f"File: {dicom_path}")
    print()
    
    # Check rescale tags
    has_slope = hasattr(plan, 'RescaleSlope')
    has_intercept = hasattr(plan, 'RescaleIntercept')
    
    print("RESCALE TAGS:")
    print(f"  RescaleSlope present: {has_slope}")
    if has_slope:
        print(f"  RescaleSlope value: {plan.RescaleSlope} (type: {type(plan.RescaleSlope)})")
    
    print(f"  RescaleIntercept present: {has_intercept}")
    if has_intercept:
        print(f"  RescaleIntercept value: {plan.RescaleIntercept} (type: {type(plan.RescaleIntercept)})")
    
    print()
    
    # Check other common tags
    print("OTHER COMMON TAGS:")
    tags_to_check = [
        'PhotometricInterpretation',
        'PresentationLUTShape', 
        'WindowCenter',
        'WindowWidth',
        'PixelIntensityRelationship',
        'BitsStored',
        'BitsAllocated'
    ]
    
    for tag in tags_to_check:
        has_tag = hasattr(plan, tag)
        print(f"  {tag}: {'✓' if has_tag else '✗'}")
        if has_tag:
            print(f"    Value: {getattr(plan, tag)}")
    
    print()
    
    # Test different scenarios
    print("SCENARIO ANALYSIS:")
    
    # Scenario 1: Both present
    if has_slope and has_intercept:
        print("✓ Both RescaleSlope and RescaleIntercept are present")
        print("  → Can apply full rescale transformation")
    
    # Scenario 2: Only slope present
    elif has_slope and not has_intercept:
        print("⚠ Only RescaleSlope is present, RescaleIntercept is missing")
        print("  → Should apply only slope transformation (intercept = 0)")
    
    # Scenario 3: Only intercept present  
    elif not has_slope and has_intercept:
        print("⚠ Only RescaleIntercept is present, RescaleSlope is missing")
        print("  → Should apply only intercept transformation (slope = 1)")
    
    # Scenario 4: Neither present
    else:
        print("✗ Neither RescaleSlope nor RescaleIntercept are present")
        print("  → No rescale transformation needed")
    
    return plan

def demonstrate_tag_handling():
    """Demonstrate proper handling of DICOM tags"""
    
    print("\n=== PROPER TAG HANDLING EXAMPLES ===")
    
    # Example 1: Current approach (requires both)
    print("CURRENT APPROACH (requires both):")
    print("if hasattr(plan, 'RescaleSlope') and hasattr(plan, 'RescaleIntercept'):")
    print("    pixel_array = pixel_array * slope + intercept")
    print("→ Only works when BOTH tags are present")
    print()
    
    # Example 2: Individual handling
    print("IMPROVED APPROACH (handles each individually):")
    print("slope = getattr(plan, 'RescaleSlope', 1.0)  # Default to 1.0 if missing")
    print("intercept = getattr(plan, 'RescaleIntercept', 0.0)  # Default to 0.0 if missing")
    print("pixel_array = pixel_array * slope + intercept")
    print("→ Works regardless of which tags are present")
    print()
    
    # Example 3: Safe conversion
    print("SAFE CONVERSION APPROACH:")
    print("def safe_float(value, default=0.0):")
    print("    try:")
    print("        return float(value)")
    print("    except (ValueError, TypeError):")
    print("        return default")
    print()
    print("slope = safe_float(getattr(plan, 'RescaleSlope', 1.0), 1.0)")
    print("intercept = safe_float(getattr(plan, 'RescaleIntercept', 0.0), 0.0)")

if __name__ == "__main__":
    # Test with your DICOM file
    dicom_path = "/home/ch215616/ww/code/llm/experiments/yolov7/create_wrist_fracture_dataset/extraction_aug_2025/dcm/26152148/DX.1.2.392.200036.9125.4.0.2535480170.369261468.3253705757"
    
    if os.path.exists(dicom_path):
        plan = analyze_dicom_tags(dicom_path)
        demonstrate_tag_handling()
    else:
        print(f"DICOM file not found: {dicom_path}")
        print("Please update the path to a valid DICOM file") 