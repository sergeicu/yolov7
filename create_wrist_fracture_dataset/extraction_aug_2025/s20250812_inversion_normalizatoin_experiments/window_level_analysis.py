import pydicom as dicom
import numpy as np

def analyze_window_level_behavior(dicom_path):
    """Analyze why window/level was producing non-black images"""
    
    plan = dicom.read_file(dicom_path)
    pixel_array = plan.pixel_array.astype(float)
    
    print("=== WINDOW/LEVEL ANALYSIS ===")
    print(f"Original pixel range: {np.min(pixel_array):.2f} to {np.max(pixel_array):.2f}")
    
    # Get window/level settings
    window_center = float(plan.WindowCenter)
    window_width = float(plan.WindowWidth)
    window_min = window_center - window_width / 2
    window_max = window_center + window_width / 2
    
    print(f"\nWindow/Level Settings:")
    print(f"  Window Center: {window_center}")
    print(f"  Window Width: {window_width}")
    print(f"  Window Min: {window_min:.2f}")
    print(f"  Window Max: {window_max:.2f}")
    
    # Analyze data distribution relative to window
    values_below_window = np.sum(pixel_array < window_min)
    values_in_window = np.sum((pixel_array >= window_min) & (pixel_array <= window_max))
    values_above_window = np.sum(pixel_array > window_max)
    total_values = pixel_array.size
    
    print(f"\nData Distribution:")
    print(f"  Values below window: {values_below_window} ({values_below_window/total_values*100:.1f}%)")
    print(f"  Values in window: {values_in_window} ({values_in_window/total_values*100:.1f}%)")
    print(f"  Values above window: {values_above_window} ({values_above_window/total_values*100:.1f}%)")
    
    # Show what happens with your approach (CLIPPING)
    print(f"\n=== YOUR APPROACH (WITH CLIPPING) ===")
    pixel_array_clipped = np.clip(pixel_array, window_min, window_max)
    print(f"After clipping: {np.min(pixel_array_clipped):.2f} to {np.max(pixel_array_clipped):.2f}")
    
    # Normalize clipped data
    pixel_array_normalized = ((pixel_array_clipped - window_min) / (window_max - window_min)) * 255
    pixel_array_normalized = np.clip(pixel_array_normalized, 0, 255)
    
    print(f"After normalization: {np.min(pixel_array_normalized):.2f} to {np.max(pixel_array_normalized):.2f}")
    
    # Show what happens WITHOUT clipping (preserves all data)
    print(f"\n=== ALTERNATIVE APPROACH (WITHOUT CLIPPING) ===")
    pixel_array_no_clip = pixel_array.copy()
    print(f"Original data: {np.min(pixel_array_no_clip):.2f} to {np.max(pixel_array_no_clip):.2f}")
    
    # Map window to 0-255, but don't clip
    pixel_array_mapped = ((pixel_array_no_clip - window_min) / (window_max - window_min)) * 255
    print(f"After mapping (no clip): {np.min(pixel_array_mapped):.2f} to {np.max(pixel_array_mapped):.2f}")
    
    # Now clip to 0-255 for PNG
    pixel_array_final = np.clip(pixel_array_mapped, 0, 255)
    print(f"After final clip: {np.min(pixel_array_final):.2f} to {np.max(pixel_array_final):.2f}")
    
    # Compare the two approaches
    print(f"\n=== COMPARISON ===")
    print(f"Your approach (with clipping):")
    print(f"  - Range: {np.min(pixel_array_normalized):.2f} to {np.max(pixel_array_normalized):.2f}")
    print(f"  - Non-zero pixels: {np.sum(pixel_array_normalized > 0)}")
    print(f"  - Black pixels: {np.sum(pixel_array_normalized == 0)}")
    
    print(f"\nAlternative approach (without clipping):")
    print(f"  - Range: {np.min(pixel_array_final):.2f} to {np.max(pixel_array_final):.2f}")
    print(f"  - Non-zero pixels: {np.sum(pixel_array_final > 0)}")
    print(f"  - Black pixels: {np.sum(pixel_array_final == 0)}")
    
    # Show why your approach works
    print(f"\n=== WHY YOUR APPROACH PRODUCES NON-BLACK IMAGES ===")
    print(f"1. Your data range: {np.min(pixel_array):.2f} to {np.max(pixel_array):.2f}")
    print(f"2. Window range: {window_min:.2f} to {window_max:.2f}")
    print(f"3. After clipping: {np.min(pixel_array_clipped):.2f} to {np.max(pixel_array_clipped):.2f}")
    print(f"4. The clipped data still has variation within the window!")
    print(f"5. This variation gets mapped to 0-255, creating visible contrast")
    
    # Show what would happen if data was completely outside window
    print(f"\n=== WHAT IF DATA WAS OUTSIDE WINDOW? ===")
    if values_in_window == 0:
        print("If ALL data was outside the window, you'd get a black image")
        print("But your data has significant overlap with the window range")
    else:
        print("Your data has good overlap with the window range")
        print("That's why you get visible images, not black ones")

if __name__ == "__main__":
    dicom_path = "/home/ch215616/ww/code/llm/experiments/yolov7/create_wrist_fracture_dataset/extraction_aug_2025/dcm/26152148/DX.1.2.392.200036.9125.4.0.2535480170.369261468.3253705757"
    analyze_window_level_behavior(dicom_path) 