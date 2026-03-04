"""
Accuracy Calculator for GROUPED Version
========================================
Nuqte ligatures ke saath hain, accuracy calculate karenge
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
    return cv2.bilateralFilter(gray_image, 9, 75, 75)


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
    """Complete enhanced preprocessing pipeline"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    denoised = bilateral_denoise(gray)
    enhanced = enhance_contrast(denoised)
    
    _, initial_binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    skew_angle = detect_skew_angle(initial_binary)
    if abs(skew_angle) > 0.5:
        enhanced = correct_skew(enhanced, skew_angle)
        print(f"  ↳ Skew corrected: {skew_angle:.2f}°")
    
    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, blockSize=13, C=3)
    
    kernel_tiny = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_tiny)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_small)
    
    return binary


def group_diacritics_with_ligatures(primary_components, secondary_components):
    """Group diacritics with their nearest primary ligature"""
    grouped = []
    used_diacritics = set()
    
    for primary in primary_components:
        p_left = primary['x']
        p_right = primary['x'] + primary['w']
        p_top = primary['y']
        p_bottom = primary['y'] + primary['h']
        p_center_x = primary['x'] + primary['w'] / 2
        
        associated_diacritics = []
        
        for idx, secondary in enumerate(secondary_components):
            if idx in used_diacritics:
                continue
            
            s_center_x = secondary['x'] + secondary['w'] / 2
            s_top = secondary['y']
            s_bottom = secondary['y'] + secondary['h']
            
            horizontal_overlap = (p_left - 15 <= s_center_x <= p_right + 15)
            vertical_distance_top = abs(s_bottom - p_top)
            vertical_distance_bottom = abs(s_top - p_bottom)
            vertical_proximity = (vertical_distance_top < 40 or vertical_distance_bottom < 40)
            
            if horizontal_overlap and vertical_proximity:
                associated_diacritics.append(secondary)
                used_diacritics.add(idx)
        
        if associated_diacritics:
            all_components = [primary] + associated_diacritics
            combined_x = min(c['x'] for c in all_components)
            combined_y = min(c['y'] for c in all_components)
            combined_max_x = max(c['x'] + c['w'] for c in all_components)
            combined_max_y = max(c['y'] + c['h'] for c in all_components)
            combined_w = combined_max_x - combined_x
            combined_h = combined_max_y - combined_y
            
            grouped.append({
                'x': combined_x, 'y': combined_y,
                'w': combined_w, 'h': combined_h,
                'area': combined_w * combined_h,
                'diacritic_count': len(associated_diacritics)
            })
        else:
            grouped.append({
                'x': primary['x'], 'y': primary['y'],
                'w': primary['w'], 'h': primary['h'],
                'area': primary['area'],
                'diacritic_count': 0
            })
    
    orphan_diacritics = [secondary_components[i] for i in range(len(secondary_components)) 
                        if i not in used_diacritics]
    
    return grouped, orphan_diacritics


def detect_grouped_characters(image_path):
    """Detect grouped ligatures (with diacritics)"""
    print(f"\n🔍 Loading image: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Could not load image!")
        return None, None, None
    
    print(f"✓ Image loaded: {image.shape}")
    
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
    min_area = 3
    
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
        return 0, 0, 0
    
    # Baseline detection
    h_proj = np.sum(binary, axis=1)
    baseline = np.argmax(h_proj)
    print(f"✓ Baseline at row: {baseline}")
    
    # Classify into primary and secondary
    heights = [c['h'] for c in components]
    median_height = np.median(heights)
    threshold_height = median_height * 0.5
    
    primary_components = []
    secondary_components = []
    
    for comp in components:
        if abs(comp['cy'] - baseline) > threshold_height:
            secondary_components.append(comp)
        else:
            primary_components.append(comp)
    
    # Group diacritics with ligatures
    grouped_ligatures, orphan_diacritics = group_diacritics_with_ligatures(
        primary_components, secondary_components
    )
    
    total_grouped = len(grouped_ligatures)
    total_diacritics_in_groups = sum(g['diacritic_count'] for g in grouped_ligatures)
    total_orphan_diacritics = len(orphan_diacritics)
    
    print(f"✓ Grouped ligatures: {total_grouped}")
    print(f"✓ Diacritics in groups: {total_diacritics_in_groups}")
    print(f"✓ Orphan diacritics: {total_orphan_diacritics}")
    
    return total_grouped, total_diacritics_in_groups, total_orphan_diacritics


def calculate_metrics(actual_ligatures, actual_orphan_diacritics,
                     detected_ligatures, detected_orphan_diacritics):
    """Calculate accuracy metrics for grouped version"""
    
    # Ligatures (with diacritics grouped)
    tp_ligatures = min(actual_ligatures, detected_ligatures)
    fp_ligatures = max(0, detected_ligatures - actual_ligatures)
    fn_ligatures = max(0, actual_ligatures - detected_ligatures)
    
    # Orphan diacritics
    tp_orphans = min(actual_orphan_diacritics, detected_orphan_diacritics)
    fp_orphans = max(0, detected_orphan_diacritics - actual_orphan_diacritics)
    fn_orphans = max(0, actual_orphan_diacritics - detected_orphan_diacritics)
    
    # Overall
    tp_total = tp_ligatures + tp_orphans
    fp_total = fp_ligatures + fp_orphans
    fn_total = fn_ligatures + fn_orphans
    
    # Metrics
    precision = (tp_total / (tp_total + fp_total)) * 100 if (tp_total + fp_total) > 0 else 0
    recall = (tp_total / (tp_total + fn_total)) * 100 if (tp_total + fn_total) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    actual_total = actual_ligatures + actual_orphan_diacritics
    accuracy = (tp_total / actual_total) * 100 if actual_total > 0 else 0
    
    return {
        'actual_ligatures': actual_ligatures,
        'actual_orphan_diacritics': actual_orphan_diacritics,
        'actual_total': actual_total,
        'detected_ligatures': detected_ligatures,
        'detected_orphan_diacritics': detected_orphan_diacritics,
        'detected_total': detected_ligatures + detected_orphan_diacritics,
        'tp_ligatures': tp_ligatures,
        'fp_ligatures': fp_ligatures,
        'fn_ligatures': fn_ligatures,
        'tp_orphans': tp_orphans,
        'fp_orphans': fp_orphans,
        'fn_orphans': fn_orphans,
        'tp_total': tp_total,
        'fp_total': fp_total,
        'fn_total': fn_total,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy
    }


def print_report(image_name, metrics):
    """Print accuracy report"""
    print("\n" + "="*70)
    print(f"📊 GROUPED VERSION ACCURACY REPORT: {image_name}")
    print("="*70)
    
    print("\n📈 COUNTS:")
    print("-"*70)
    print(f"{'Category':<30} {'Actual':<15} {'Detected':<15} {'Match'}")
    print("-"*70)
    
    ligatures_match = "✓" if metrics['actual_ligatures'] == metrics['detected_ligatures'] else "✗"
    orphans_match = "✓" if metrics['actual_orphan_diacritics'] == metrics['detected_orphan_diacritics'] else "✗"
    
    print(f"{'Grouped Ligatures':<30} {metrics['actual_ligatures']:<15} {metrics['detected_ligatures']:<15} {ligatures_match}")
    print(f"{'Orphan Diacritics':<30} {metrics['actual_orphan_diacritics']:<15} {metrics['detected_orphan_diacritics']:<15} {orphans_match}")
    print(f"{'TOTAL':<30} {metrics['actual_total']:<15} {metrics['detected_total']:<15}")
    
    print("\n" + "="*70)
    print("🔍 CONFUSION MATRIX (Grouped Ligatures):")
    print("-"*70)
    print(f"  ✅ True Positive (TP):   {metrics['tp_ligatures']}")
    print(f"  ❌ False Positive (FP):  {metrics['fp_ligatures']}")
    print(f"  ⚠️  False Negative (FN):  {metrics['fn_ligatures']}")
    
    print("\n🔍 CONFUSION MATRIX (Orphan Diacritics):")
    print("-"*70)
    print(f"  ✅ True Positive (TP):   {metrics['tp_orphans']}")
    print(f"  ❌ False Positive (FP):  {metrics['fp_orphans']}")
    print(f"  ⚠️  False Negative (FN):  {metrics['fn_orphans']}")
    
    print("\n" + "="*70)
    print("📊 OVERALL CONFUSION MATRIX:")
    print("-"*70)
    print(f"  ✅ True Positive (TP):   {metrics['tp_total']:<10} (Correctly detected)")
    print(f"  ❌ False Positive (FP):  {metrics['fp_total']:<10} (Over-detected)")
    print(f"  ⚠️  False Negative (FN):  {metrics['fn_total']:<10} (Missed)")
    
    print("\n" + "="*70)
    print("📈 ACCURACY METRICS:")
    print("="*70)
    
    print(f"\n  🎯 PRECISION: {metrics['precision']:.2f}%")
    print(f"     (What % of detected are correct)")
    print(f"     Formula: TP/(TP+FP) = {metrics['tp_total']}/({metrics['tp_total']}+{metrics['fp_total']})")
    
    print(f"\n  🎯 RECALL: {metrics['recall']:.2f}%")
    print(f"     (What % of actual were detected)")
    print(f"     Formula: TP/(TP+FN) = {metrics['tp_total']}/({metrics['tp_total']}+{metrics['fn_total']})")
    
    print(f"\n  🎯 F1-SCORE: {metrics['f1_score']:.2f}%")
    print(f"     (Balance between Precision and Recall)")
    
    print(f"\n  🎯 ACCURACY: {metrics['accuracy']:.2f}%")
    print(f"     (Overall correctness)")
    print(f"     Formula: TP/Total = {metrics['tp_total']}/{metrics['actual_total']}")
    
    print("\n" + "="*70)
    print("✅ FINAL RESULT:")
    print("="*70)
    print("┌─────────────────────────────────────┐")
    print(f"│ Precision:  {metrics['precision']:>6.2f}%              │")
    print(f"│ Recall:     {metrics['recall']:>6.2f}%              │")
    print(f"│ F1-Score:   {metrics['f1_score']:>6.2f}%              │")
    print(f"│ Accuracy:   {metrics['accuracy']:>6.2f}%              │")
    print("└─────────────────────────────────────┘")


def main():
    print("="*70)
    print("GROUPED VERSION ACCURACY CALCULATOR")
    print("="*70)
    
    if len(sys.argv) < 3:
        print("\n📝 Usage:")
        print("   python grouped_accuracy.py <image_path> <actual_ligatures> <actual_orphan_diacritics>")
        print("\n📌 Example:")
        print('   python grouped_accuracy.py "input_images/4 (124).png" 35 30')
        print("\n❓ Parameters:")
        print("   1. actual_ligatures: Kitne ligatures hain (diacritics ke saath counted)")
        print("   2. actual_orphan_diacritics: Kitne diacritics alag hain (kisi ligature ke saath nahi)")
        return
    
    image_path = sys.argv[1]
    
    try:
        actual_ligatures = int(sys.argv[2])
        actual_orphan_diacritics = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    except:
        print("❌ Error: Please provide numbers")
        return
    
    if not os.path.exists(image_path):
        print(f"❌ Error: File not found: {image_path}")
        return
    
    print(f"\n📋 Settings:")
    print(f"   Image: {image_path}")
    print(f"   Actual Grouped Ligatures: {actual_ligatures}")
    print(f"   Actual Orphan Diacritics: {actual_orphan_diacritics}")
    
    # Detect
    print("\n🔄 Detecting characters...")
    detected_ligatures, detected_diacritics_in_groups, detected_orphan_diacritics = detect_grouped_characters(image_path)
    
    if detected_ligatures is None:
        print("❌ Detection failed!")
        return
    
    print(f"\n✓ Detected Grouped Ligatures: {detected_ligatures}")
    print(f"✓ Detected Orphan Diacritics: {detected_orphan_diacritics}")
    
    # Calculate metrics
    print("\n🧮 Calculating metrics...")
    metrics = calculate_metrics(
        actual_ligatures, actual_orphan_diacritics,
        detected_ligatures, detected_orphan_diacritics
    )
    
    # Print report
    image_name = os.path.basename(image_path)
    print_report(image_name, metrics)
    
    # Save results
    os.makedirs("evaluation_results", exist_ok=True)
    base_name = os.path.splitext(image_name)[0]
    result_file = f"evaluation_results/{base_name}_grouped_accuracy.json"
    
    with open(result_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✅ Results saved to: {result_file}")


if __name__ == "__main__":
    main()
