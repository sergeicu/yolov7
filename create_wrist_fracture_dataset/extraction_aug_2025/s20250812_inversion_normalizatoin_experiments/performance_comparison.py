import time
import os
import png
import pydicom as dicom
import numpy as np

def original_approach(mri_file):
    """Original inefficient approach with list operations"""
    plan = dicom.read_file(mri_file)
    shape = plan.pixel_array.shape

    # Inefficient: Convert to list
    image_2d = []
    max_val = 0
    for row in plan.pixel_array:
        pixels = []
        for col in row:
            pixels.append(col)
            if col > max_val: max_val = col
        image_2d.append(pixels)

    # Inefficient: Convert to numpy, then back to list
    pixel_array = np.array(image_2d)
    # Apply inversion
    if hasattr(plan, 'PhotometricInterpretation') and plan.PhotometricInterpretation == 'MONOCHROME1':
        pixel_array = np.max(pixel_array) - pixel_array
    image_2d = pixel_array.tolist()
    max_val = max(max(row) for row in image_2d)

    # Inefficient: Manual normalization with loops
    image_2d_scaled = []
    for row in image_2d:
        row_scaled = []
        for col in row:
            col_scaled = int((float(col) / float(max_val)) * 255.0)
            row_scaled.append(col_scaled)
        image_2d_scaled.append(row_scaled)
    
    return image_2d_scaled

def optimized_approach(mri_file):
    """Optimized approach with pure numpy operations"""
    plan = dicom.read_file(mri_file)
    
    # Efficient: Direct numpy array
    pixel_array = plan.pixel_array.astype(float)
    
    # Efficient: Pure numpy operations
    if hasattr(plan, 'PhotometricInterpretation') and plan.PhotometricInterpretation == 'MONOCHROME1':
        pixel_array = np.max(pixel_array) - pixel_array
    
    # Efficient: Vectorized normalization
    pixel_min = np.min(pixel_array)
    pixel_max = np.max(pixel_array)
    if pixel_max > pixel_min:
        pixel_array = ((pixel_array - pixel_min) / (pixel_max - pixel_min)) * 255
    pixel_array = np.clip(pixel_array, 0, 255).astype(np.uint8)
    
    # Only convert to list at the very end (required by PNG library)
    return pixel_array.tolist()

def compare_performance(dicom_path, num_runs=5):
    """Compare performance between original and optimized approaches"""
    
    print("=== PERFORMANCE COMPARISON ===")
    print(f"Testing with: {dicom_path}")
    print(f"Number of runs: {num_runs}")
    print()
    
    # Test original approach
    print("Testing ORIGINAL approach (inefficient)...")
    original_times = []
    for i in range(num_runs):
        start_time = time.time()
        result_original = original_approach(dicom_path)
        end_time = time.time()
        original_times.append(end_time - start_time)
        print(f"  Run {i+1}: {original_times[-1]:.4f} seconds")
    
    # Test optimized approach
    print("\nTesting OPTIMIZED approach (efficient)...")
    optimized_times = []
    for i in range(num_runs):
        start_time = time.time()
        result_optimized = optimized_approach(dicom_path)
        end_time = time.time()
        optimized_times.append(end_time - start_time)
        print(f"  Run {i+1}: {optimized_times[-1]:.4f} seconds")
    
    # Calculate statistics
    original_avg = np.mean(original_times)
    original_std = np.std(original_times)
    optimized_avg = np.mean(optimized_times)
    optimized_std = np.std(optimized_times)
    
    speedup = original_avg / optimized_avg
    
    print(f"\n=== RESULTS ===")
    print(f"Original approach:")
    print(f"  Average time: {original_avg:.4f} ± {original_std:.4f} seconds")
    print(f"  Total time: {sum(original_times):.4f} seconds")
    
    print(f"\nOptimized approach:")
    print(f"  Average time: {optimized_avg:.4f} ± {optimized_std:.4f} seconds")
    print(f"  Total time: {sum(optimized_times):.4f} seconds")
    
    print(f"\nPerformance improvement:")
    print(f"  Speedup: {speedup:.2f}x faster")
    print(f"  Time saved: {original_avg - optimized_avg:.4f} seconds per file")
    print(f"  Efficiency gain: {(1 - optimized_avg/original_avg)*100:.1f}%")
    
    # Verify results are identical
    if result_original == result_optimized:
        print(f"\n✅ Results are identical - optimization is safe!")
    else:
        print(f"\n❌ Results differ - check optimization logic!")
    
    return {
        'original_times': original_times,
        'optimized_times': optimized_times,
        'speedup': speedup,
        'results_identical': result_original == result_optimized
    }

def analyze_inefficiencies():
    """Analyze why the original approach is inefficient"""
    
    print("\n=== INEFFICIENCY ANALYSIS ===")
    print("Original approach problems:")
    print("1. ❌ Manual list building with nested loops")
    print("   - O(n²) complexity for finding max value")
    print("   - Python loops are slow")
    print()
    print("2. ❌ Unnecessary conversions:")
    print("   - DICOM → List → Numpy → List")
    print("   - Each conversion copies data")
    print()
    print("3. ❌ Manual normalization with loops:")
    print("   - O(n²) complexity for scaling")
    print("   - No vectorization benefits")
    print()
    print("Optimized approach improvements:")
    print("1. ✅ Direct numpy array access:")
    print("   - O(n) complexity for operations")
    print("   - No unnecessary conversions")
    print()
    print("2. ✅ Vectorized operations:")
    print("   - Single numpy operation for entire array")
    print("   - C-optimized under the hood")
    print()
    print("3. ✅ Minimal conversions:")
    print("   - DICOM → Numpy → List (only at end)")
    print("   - List conversion only when required by PNG library")

if __name__ == "__main__":
    dicom_path = "/home/ch215616/ww/code/llm/experiments/yolov7/create_wrist_fracture_dataset/extraction_aug_2025/dcm/26152148/DX.1.2.392.200036.9125.4.0.2535480170.369261468.3253705757"
    
    if os.path.exists(dicom_path):
        results = compare_performance(dicom_path, num_runs=3)
        analyze_inefficiencies()
    else:
        print(f"DICOM file not found: {dicom_path}")
        print("Please update the path to a valid DICOM file") 