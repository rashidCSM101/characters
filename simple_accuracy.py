"""
Simple Accuracy Calculator (ENHANCED)
======================================
With Enhanced Preprocessing:
1. Contrast Enhancement (CLAHE)
2. Adaptive Thresholding
3. Noise Removal (Morphological Operations)
4. Skew Correction
"""

import cv2
import numpy as np
import os
import sys
import json


def enhance_contrast(gray_image):
    """CLAHE - Contrast Enhancement"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_image)


def bilateral_denoise(gray_image):
    """Bilateral filter for edge-preserving denoising"""
    denoised = cv2.bilateralFilter(gray_image, 9, 75, 75)
    return denoised


def detect_skew_angle(binary_image):
    """Detect skew angle using Hough Transform"""
    edges = cv2.Canny(binary_image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    
    if lines is None:
        return 0.0
    
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 != 0:
            angle = np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi
            if -45 < angle < 45:
                angles.append(angle)
    
    return np.median(angles) if angles else 0.0


def correct_skew(image, angle):
    """Rotate image to correct skew"""
    if abs(angle) < 0.5:
        return image
    
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    
    return cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def enhanced_preprocessing(image):
    """Complete enhanced preprocessing pipeline - IMPROVED"""
    # 1. Grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 1.5. IMPROVED - Bilateral filter denoising
    denoised = bilateral_denoise(gray)
    
    # 2. Contrast Enhancement (CLAHE)
    enhanced = enhance_contrast(denoised)
    
    # 3. Initial binarization for skew detection
    _, initial_binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 4. Skew Correction
    skew_angle = detect_skew_angle(initial_binary)
    if abs(skew_angle) > 0.5:
        enhanced = correct_skew(enhanced, skew_angle)
        print(f"  ↳ Skew corrected: {skew_angle:.2f}°")
    
    # 5. IMPROVED - Optimized Adaptive Thresholding
    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, blockSize=13, C=3)
    
    # 6. IMPROVED - Gentler Noise Removal (Morphological Operations)
    kernel_tiny = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_tiny)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_small)
    
    return binary


def detect_characters(image_path):
    """Detect primary ligatures and diacritics with ENHANCED preprocessing"""
    print(f"\n🔍 Loading image: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Could not load image!")
        return None, None
    
    print(f"✓ Image loaded: {image.shape}")
    
    # ENHANCED Preprocessing
    print("📌 ENHANCED PREPROCESSING:")
    print("  ↳ Contrast Enhancement (CLAHE)...")
    print("  ↳ Adaptive Thresholding...")
    print("  ↳ Noise Removal (Morphological Ops)...")
    binary = enhanced_preprocessing(image)
    print("✓ Enhanced binarization done")
    
    # Connected component labeling
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    print(f"✓ Found {num_labels - 1} components")
    
    components = []
    min_area = 3  # IMPROVED: Reduced from 10 to 3
    
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        cy = centroids[i][1]
        
        if area >= min_area:
            components.append({
                'x': x, 'y': y, 'w': w, 'h': h, 'area': area, 'cy': cy
            })
    
    if not components:
        print("❌ No components found!")
        return 0, 0
    
    # IMPROVED: Robust baseline detection
    # Method 1: Traditional horizontal projection
    h_proj = np.sum(binary, axis=1)
    baseline_proj = np.argmax(h_proj)
    
    # Method 2: Contour-based (more robust)
    contours, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        bottom_points = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            bottom_points.append(y + h)
        baseline_contour = int(np.median(bottom_points))
        baseline = int((baseline_proj + baseline_contour) / 2)
    else:
        baseline = baseline_proj
    
    print(f"✓ Baseline at row: {baseline}")
    
    # IMPROVED: Smarter primary/diacritic separation with multiple criteria
    avg_height = np.mean([c['h'] for c in components])
    avg_area = np.mean([c['area'] for c in components])
    avg_width = np.mean([c['w'] for c in components])
    
    primary_count = 0
    diacritics_count = 0
    
    for comp in components:
        y_top = comp['y']
        y_bottom = comp['y'] + comp['h']
        
        # Criterion 1: Baseline touching
        touches_baseline = y_top <= baseline <= y_bottom
        
        # Criterion 2: Size relative to average
        relative_area = comp['area'] / avg_area
        relative_height = comp['h'] / avg_height
        
        # Criterion 3: Aspect ratio (diacritics tend to be circular)
        aspect_ratio = comp['w'] / comp['h'] if comp['h'] > 0 else 1
        is_round = 0.6 < aspect_ratio < 1.4
        
        # Criterion 4: Distance from baseline
        distance_from_baseline = min(abs(y_top - baseline), abs(y_bottom - baseline))
        is_far_from_baseline = distance_from_baseline > avg_height * 0.3
        
        # Classification: Diacritics are small, round, and far from baseline
        is_diacritic = (
            relative_area < 0.25 and 
            is_far_from_baseline and 
            is_round
        ) or (
            relative_height < 0.4 and 
            relative_area < 0.2
        )
        
        if not is_diacritic and (touches_baseline or relative_height > 0.45):
            primary_count += 1
        else:
            diacritics_count += 1
    
    return primary_count, diacritics_count


def calculate_metrics(actual_primary, actual_diacritics, detected_primary, detected_diacritics):
    """Calculate accuracy metrics"""
    
    actual_total = actual_primary + actual_diacritics
    detected_total = detected_primary + detected_diacritics
    
    # For primary ligatures
    tp_primary = min(detected_primary, actual_primary)
    fp_primary = max(0, detected_primary - actual_primary)
    fn_primary = max(0, actual_primary - detected_primary)
    
    # For diacritics
    tp_diacritics = min(detected_diacritics, actual_diacritics)
    fp_diacritics = max(0, detected_diacritics - actual_diacritics)
    fn_diacritics = max(0, actual_diacritics - detected_diacritics)
    
    # Overall
    tp_total = tp_primary + tp_diacritics
    fp_total = fp_primary + fp_diacritics
    fn_total = fn_primary + fn_diacritics
    
    # Calculate metrics
    if (tp_total + fp_total) > 0:
        precision = tp_total / (tp_total + fp_total)
    else:
        precision = 0
    
    if (tp_total + fn_total) > 0:
        recall = tp_total / (tp_total + fn_total)
    else:
        recall = 0
    
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0
    
    if actual_total > 0:
        accuracy = tp_total / actual_total
    else:
        accuracy = 0
    
    return {
        'actual_primary': actual_primary,
        'actual_diacritics': actual_diacritics,
        'actual_total': actual_total,
        'detected_primary': detected_primary,
        'detected_diacritics': detected_diacritics,
        'detected_total': detected_total,
        'tp_primary': tp_primary,
        'fp_primary': fp_primary,
        'fn_primary': fn_primary,
        'tp_diacritics': tp_diacritics,
        'fp_diacritics': fp_diacritics,
        'fn_diacritics': fn_diacritics,
        'tp_total': tp_total,
        'fp_total': fp_total,
        'fn_total': fn_total,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy
    }


def print_report(image_name, metrics):
    """Print nice report"""
    
    print("\n" + "="*70)
    print(f"📊 ACCURACY REPORT: {image_name}")
    print("="*70)
    
    print("\n📈 COUNTS:")
    print("-"*70)
    print(f"{'Category':<25} {'Actual':<15} {'Detected':<15} {'Match':<15}")
    print("-"*70)
    
    prim_match = "✓" if metrics['detected_primary'] == metrics['actual_primary'] else "✗"
    diac_match = "✓" if metrics['detected_diacritics'] == metrics['actual_diacritics'] else "✗"
    
    print(f"{'Primary Ligatures':<25} {metrics['actual_primary']:<15} {metrics['detected_primary']:<15} {prim_match:<15}")
    print(f"{'Diacritics (Dots)':<25} {metrics['actual_diacritics']:<15} {metrics['detected_diacritics']:<15} {diac_match:<15}")
    print(f"{'TOTAL':<25} {metrics['actual_total']:<15} {metrics['detected_total']:<15}")
    
    print("\n" + "="*70)
    print("🔍 CONFUSION MATRIX (Primary Ligatures):")
    print("-"*70)
    print(f"  ✅ True Positive (TP):   {metrics['tp_primary']}")
    print(f"  ❌ False Positive (FP):  {metrics['fp_primary']}")
    print(f"  ⚠️  False Negative (FN):  {metrics['fn_primary']}")
    
    print("\n🔍 CONFUSION MATRIX (Diacritics):")
    print("-"*70)
    print(f"  ✅ True Positive (TP):   {metrics['tp_diacritics']}")
    print(f"  ❌ False Positive (FP):  {metrics['fp_diacritics']}")
    print(f"  ⚠️  False Negative (FN):  {metrics['fn_diacritics']}")
    
    print("\n" + "="*70)
    print("📊 OVERALL CONFUSION MATRIX:")
    print("-"*70)
    print(f"  ✅ True Positive (TP):   {metrics['tp_total']:<10} (Correctly detected)")
    print(f"  ❌ False Positive (FP):  {metrics['fp_total']:<10} (Over-detected)")
    print(f"  ⚠️  False Negative (FN):  {metrics['fn_total']:<10} (Missed)")
    
    print("\n" + "="*70)
    print("📈 ACCURACY METRICS:")
    print("="*70)
    
    print(f"\n  🎯 PRECISION: {metrics['precision']*100:.2f}%")
    print(f"     (What % of detected are correct)")
    print(f"     Formula: TP/(TP+FP) = {metrics['tp_total']}/({metrics['tp_total']}+{metrics['fp_total']})")
    
    print(f"\n  🎯 RECALL: {metrics['recall']*100:.2f}%")
    print(f"     (What % of actual were detected)")
    print(f"     Formula: TP/(TP+FN) = {metrics['tp_total']}/({metrics['tp_total']}+{metrics['fn_total']})")
    
    print(f"\n  🎯 F1-SCORE: {metrics['f1_score']*100:.2f}%")
    print(f"     (Balance between Precision and Recall)")
    
    print(f"\n  🎯 ACCURACY: {metrics['accuracy']*100:.2f}%")
    print(f"     (Overall correctness)")
    print(f"     Formula: TP/Total = {metrics['tp_total']}/{metrics['actual_total']}")
    
    print("\n" + "="*70)
    print("✅ FINAL RESULT:")
    print("="*70)
    print(f"┌─────────────────────────────────────┐")
    print(f"│ Precision:  {metrics['precision']*100:>6.2f}%              │")
    print(f"│ Recall:     {metrics['recall']*100:>6.2f}%              │")
    print(f"│ F1-Score:   {metrics['f1_score']*100:>6.2f}%              │")
    print(f"│ Accuracy:   {metrics['accuracy']*100:>6.2f}%              │")
    print(f"└─────────────────────────────────────┘")


def main():
    print("\n" + "="*70)
    print("URDU OCR ACCURACY CALCULATOR")
    print("="*70)
    
    if len(sys.argv) < 4:
        print("\n📝 Usage:")
        print("   python simple_accuracy.py <image_path> <actual_primary> <actual_diacritics>")
        print("\n📌 Example:")
        print('   python simple_accuracy.py "input_images/4 (106).png" 650 400')
        print("\n❓ How to get actual counts:")
        print("   1. Khud manually count karo image mein")
        print("   2. Primary ligatures (main characters): count karo")
        print("   3. Diacritics (dots/nuqte): count karo")
        return
    
    image_path = sys.argv[1]
    
    try:
        actual_primary = int(sys.argv[2])
        actual_diacritics = int(sys.argv[3])
    except:
        print("❌ Error: Please provide numbers for primary and diacritics")
        return
    
    if not os.path.exists(image_path):
        print(f"❌ Error: File not found: {image_path}")
        return
    
    print(f"\n📋 Settings:")
    print(f"   Image: {image_path}")
    print(f"   Actual Primary: {actual_primary}")
    print(f"   Actual Diacritics: {actual_diacritics}")
    
    # Detect
    print("\n🔄 Detecting characters...")
    detected_primary, detected_diacritics = detect_characters(image_path)
    
    if detected_primary is None:
        print("❌ Detection failed!")
        return
    
    print(f"✓ Detected Primary: {detected_primary}")
    print(f"✓ Detected Diacritics: {detected_diacritics}")
    
    # Calculate metrics
    print("\n🧮 Calculating metrics...")
    metrics = calculate_metrics(
        actual_primary, actual_diacritics,
        detected_primary, detected_diacritics
    )
    
    # Print report
    image_name = os.path.basename(image_path)
    print_report(image_name, metrics)
    
    # Save results
    os.makedirs("evaluation_results", exist_ok=True)
    base_name = os.path.splitext(image_name)[0]
    result_file = f"evaluation_results/{base_name}_accuracy.json"
    
    with open(result_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✅ Results saved to: {result_file}")


if __name__ == "__main__":
    main()
