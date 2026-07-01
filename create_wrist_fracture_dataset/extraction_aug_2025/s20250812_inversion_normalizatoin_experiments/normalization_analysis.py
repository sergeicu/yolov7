import pydicom as dicom
import numpy as np
import matplotlib.pyplot as plt

def analyze_information_loss(dicom_path):
    """Analyze information loss in different normalization methods"""
    
    plan = dicom.read_file(dicom_path)
    pixel_array = plan.pixel_array.astype(float)
    
    print("=== INFORMATION LOSS ANALYSIS FOR ML DATASETS ===")
    print(f"Original pixel range: {np.min(pixel_array):.2f} to {np.max(pixel_array):.2f}")
    print(f"Original data type: {pixel_array.dtype}")
    print(f"Image dimensions: {pixel_array.shape}")
    
    # Get window/level settings
    window_center = float(plan.WindowCenter)
    window_width = float(plan.WindowWidth)
    window_min = window_center - window_width / 2
    window_max = window_center + window_width / 2
    
    print(f"\nWindow/Level Settings:")
    print(f"  Window Center: {window_center}")
    print(f"  Window Width: {window_width}")
    print(f"  Window Range: {window_min:.2f} to {window_max:.2f}")
    
    # Analyze data distribution
    values_below_window = np.sum(pixel_array < window_min)
    values_in_window = np.sum((pixel_array >= window_min) & (pixel_array <= window_max))
    values_above_window = np.sum(pixel_array > window_max)
    total_values = pixel_array.size
    
    print(f"\nData Distribution:")
    print(f"  Values below window: {values_below_window} ({values_below_window/total_values*100:.1f}%)")
    print(f"  Values in window: {values_in_window} ({values_in_window/total_values*100:.1f}%)")
    print(f"  Values above window: {values_above_window} ({values_above_window/total_values*100:.1f}%)")
    
    # Method 1: Window/Level Normalization
    print(f"\n=== WINDOW/LEVEL NORMALIZATION ===")
    pixel_array_window = np.clip(pixel_array, window_min, window_max)
    pixel_array_window_norm = ((pixel_array_window - window_min) / (window_max - window_min)) * 255
    pixel_array_window_norm = np.clip(pixel_array_window_norm, 0, 255).astype(np.uint8)
    
    print(f"After window/level normalization:")
    print(f"  Range: {np.min(pixel_array_window_norm)} to {np.max(pixel_array_window_norm)}")
    print(f"  Data type: {pixel_array_window_norm.dtype}")
    print(f"  Unique values: {len(np.unique(pixel_array_window_norm))}")
    
    # Calculate information loss for window/level
    original_unique = len(np.unique(pixel_array))
    window_unique = len(np.unique(pixel_array_window_norm))
    window_loss_percent = ((original_unique - window_unique) / original_unique) * 100
    
    print(f"  Information loss: {window_loss_percent:.1f}% of unique values lost")
    
    # Method 2: Min-Max Normalization
    print(f"\n=== MIN-MAX NORMALIZATION ===")
    pixel_min = np.min(pixel_array)
    pixel_max = np.max(pixel_array)
    pixel_array_minmax = ((pixel_array - pixel_min) / (pixel_max - pixel_min)) * 255
    pixel_array_minmax = np.clip(pixel_array_minmax, 0, 255).astype(np.uint8)
    
    print(f"After min-max normalization:")
    print(f"  Range: {np.min(pixel_array_minmax)} to {np.max(pixel_array_minmax)}")
    print(f"  Data type: {pixel_array_minmax.dtype}")
    print(f"  Unique values: {len(np.unique(pixel_array_minmax))}")
    
    # Calculate information loss for min-max
    minmax_unique = len(np.unique(pixel_array_minmax))
    minmax_loss_percent = ((original_unique - minmax_unique) / original_unique) * 100
    
    print(f"  Information loss: {minmax_loss_percent:.1f}% of unique values lost")
    
    # Method 3: Percentile-based normalization (robust to outliers)
    print(f"\n=== PERCENTILE NORMALIZATION ===")
    p1 = np.percentile(pixel_array, 1)
    p99 = np.percentile(pixel_array, 99)
    pixel_array_percentile = ((pixel_array - p1) / (p99 - p1)) * 255
    pixel_array_percentile = np.clip(pixel_array_percentile, 0, 255).astype(np.uint8)
    
    print(f"After percentile normalization:")
    print(f"  Range: {np.min(pixel_array_percentile)} to {np.max(pixel_array_percentile)}")
    print(f"  Data type: {pixel_array_percentile.dtype}")
    print(f"  Unique values: {len(np.unique(pixel_array_percentile))}")
    
    percentile_unique = len(np.unique(pixel_array_percentile))
    percentile_loss_percent = ((original_unique - percentile_unique) / original_unique) * 100
    
    print(f"  Information loss: {percentile_loss_percent:.1f}% of unique values lost")
    
    # Compare methods
    print(f"\n=== COMPARISON FOR ML DATASETS ===")
    print(f"Method                    | Unique Values | Information Loss | ML Suitability")
    print(f"--------------------------|---------------|------------------|---------------")
    print(f"Original                  | {original_unique:>13} | {0:>16.1f}% | Reference")
    print(f"Window/Level              | {window_unique:>13} | {window_loss_percent:>16.1f}% | Clinical")
    print(f"Min-Max                   | {minmax_unique:>13} | {minmax_loss_percent:>16.1f}% | General")
    print(f"Percentile                | {percentile_unique:>13} | {percentile_loss_percent:>16.1f}% | Robust")
    
    # ML-specific recommendations
    print(f"\n=== ML DATASET RECOMMENDATIONS ===")
    
    print(f"1. WINDOW/LEVEL NORMALIZATION:")
    print(f"   ✅ PROS:")
    print(f"      - Preserves clinical interpretation")
    print(f"      - Uses manufacturer's recommended settings")
    print(f"      - Consistent across similar image types")
    print(f"   ❌ CONS:")
    print(f"      - Loses data outside window ({values_below_window + values_above_window} pixels)")
    print(f"      - May lose subtle features")
    print(f"      - Window settings may not be optimal for ML")
    
    print(f"\n2. MIN-MAX NORMALIZATION:")
    print(f"   ✅ PROS:")
    print(f"      - Preserves ALL data")
    print(f"      - No information loss")
    print(f"      - Simple and consistent")
    print(f"      - Works well for ML training")
    print(f"   ❌ CONS:")
    print(f"      - May not match clinical display")
    print(f"      - Sensitive to outliers")
    print(f"      - Contrast may not be optimal")
    
    print(f"\n3. PERCENTILE NORMALIZATION:")
    print(f"   ✅ PROS:")
    print(f"      - Robust to outliers")
    print(f"      - Preserves most data")
    print(f"      - Good for ML training")
    print(f"   ❌ CONS:")
    print(f"      - Loses extreme values")
    print(f"      - May not match clinical display")
    
    # Final recommendation
    print(f"\n=== FINAL RECOMMENDATION FOR ML ===")
    print(f"For machine learning model training:")
    print(f"1. PRIMARY: Min-Max normalization (preserves all information)")
    print(f"2. ALTERNATIVE: Percentile normalization (robust to outliers)")
    print(f"3. AVOID: Window/Level for ML training (loses information)")
    print(f"\nFor clinical applications:")
    print(f"1. PRIMARY: Window/Level normalization (clinical accuracy)")
    print(f"2. ALTERNATIVE: Min-Max normalization (full data preservation)")
    
    return {
        'original': pixel_array,
        'window_level': pixel_array_window_norm,
        'min_max': pixel_array_minmax,
        'percentile': pixel_array_percentile
    }

def create_histogram_comparison(results):
    """Create histograms to visualize information distribution"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Information Distribution Comparison', fontsize=16)
    
    # Original data
    axes[0, 0].hist(results['original'].flatten(), bins=50, alpha=0.7, color='blue')
    axes[0, 0].set_title('Original Data')
    axes[0, 0].set_xlabel('Pixel Value')
    axes[0, 0].set_ylabel('Frequency')
    
    # Window/Level
    axes[0, 1].hist(results['window_level'].flatten(), bins=50, alpha=0.7, color='red')
    axes[0, 1].set_title('Window/Level Normalization')
    axes[0, 1].set_xlabel('Pixel Value')
    axes[0, 1].set_ylabel('Frequency')
    
    # Min-Max
    axes[1, 0].hist(results['min_max'].flatten(), bins=50, alpha=0.7, color='green')
    axes[1, 0].set_title('Min-Max Normalization')
    axes[1, 0].set_xlabel('Pixel Value')
    axes[1, 0].set_ylabel('Frequency')
    
    # Percentile
    axes[1, 1].hist(results['percentile'].flatten(), bins=50, alpha=0.7, color='orange')
    axes[1, 1].set_title('Percentile Normalization')
    axes[1, 1].set_xlabel('Pixel Value')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('normalization_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nHistogram comparison saved as 'normalization_comparison.png'")

if __name__ == "__main__":
    dicom_path = "/home/ch215616/ww/code/llm/experiments/yolov7/create_wrist_fracture_dataset/extraction_aug_2025/dcm/26152148/DX.1.2.392.200036.9125.4.0.2535480170.369261468.3253705757"
    
    results = analyze_information_loss(dicom_path)
    
    # Create visualization
    try:
        create_histogram_comparison(results)
    except ImportError:
        print("\nMatplotlib not available - skipping histogram generation") 