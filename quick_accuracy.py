"""
Quick Accuracy Evaluation
=========================
Usage: python quick_accuracy.py <image_path> <actual_primary> <actual_diacritics>

Example:
    python quick_accuracy.py input_images/4.png 650 400
"""

import cv2
import numpy as np
import os
import sys
import json
from datetime import datetime


def preprocess_binarization(image):
    """Binarization using Otsu's thresholding"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def detect_and_separate(image_path):
    """Detect and separate primary/diacritics"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load {image_path}")
        return None, None
    
    binary = preprocess_binarization(image)
    
    # Connected component labeling
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    components = []
    min_area = 10
    
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        if area >= min_area:
            components.append({'x': x, 'y': y, 'width': w, 'height': h, 'area': area})
    
    if not components:
        return 0, 0
    
    # Baseline detection
    baseline_y = np.argmax(np.sum(binary, axis=1))
    
    # Separate
    avg_height = np.mean([c['height'] for c in components])
    avg_area = np.mean([c['area'] for c in components])
    
    primary = 0
    diacritics = 0
    
    for comp in components:
        y_top = comp['y']
        y_bottom = comp['y'] + comp['height']
        touches_baseline = y_top <= baseline_y <= y_bottom
        is_small = comp['area'] < avg_area * 0.3 or comp['height'] < avg_height * 0.5
        
        if touches_baseline or (not is_small and comp['height'] > avg_height * 0.4):
            primary += 1
        else:
            diacritics += 1
    
    return primary, diacritics


def calculate_accuracy(actual_primary, actual_diacritics, detected_primary, detected_diacritics):
    """Calculate all accuracy metrics"""
    
    actual_total = actual_primary + actual_diacritics
    detected_total = detected_primary + detected_diacritics
    
    # Primary Metrics
    tp_primary = min(detected_primary, actual_primary)
    fp_primary = max(0, detected_primary - actual_primary)
    fn_primary = max(0, actual_primary - detected_primary)
    
    # Diacritics Metrics
    tp_diacritics = min(detected_diacritics, actual_diacritics)
    fp_diacritics = max(0, detected_diacritics - actual_diacritics)
    fn_diacritics = max(0, actual_diacritics - detected_diacritics)
    
    # Overall
    tp_total = tp_primary + tp_diacritics
    fp_total = fp_primary + fp_diacritics
    fn_total = fn_primary + fn_diacritics
    tn_total = 0  # Not applicable for detection
    
    # Metrics calculation
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = tp_total / actual_total if actual_total > 0 else 0
    
    return {
        'actual': {'primary': actual_primary, 'diacritics': actual_diacritics, 'total': actual_total},
        'detected': {'primary': detected_primary, 'diacritics': detected_diacritics, 'total': detected_total},
        'confusion_matrix': {'TP': tp_total, 'FP': fp_total, 'FN': fn_total, 'TN': tn_total},
        'primary': {'TP': tp_primary, 'FP': fp_primary, 'FN': fn_primary},
        'diacritics': {'TP': tp_diacritics, 'FP': fp_diacritics, 'FN': fn_diacritics},
        'metrics': {'precision': precision, 'recall': recall, 'f1_score': f1_score, 'accuracy': accuracy}
    }


def print_results(results, image_name):
    """Print formatted results"""
    
    print("\n" + "="*70)
    print(f"ACCURACY EVALUATION RESULTS: {image_name}")
    print("="*70)
    
    # Detection Counts
    print("\n📊 DETECTION COUNTS:")
    print("-"*50)
    print(f"{'Category':<20} {'Actual':<15} {'Detected':<15}")
    print("-"*50)
    print(f"{'Primary Ligatures':<20} {results['actual']['primary']:<15} {results['detected']['primary']:<15}")
    print(f"{'Diacritics':<20} {results['actual']['diacritics']:<15} {results['detected']['diacritics']:<15}")
    print(f"{'TOTAL':<20} {results['actual']['total']:<15} {results['detected']['total']:<15}")
    
    # Confusion Matrix - Overall
    print("\n📊 CONFUSION MATRIX (Overall):")
    print("-"*50)
    cm = results['confusion_matrix']
    print(f"  True Positive (TP):   {cm['TP']:<10} (Correctly detected)")
    print(f"  False Positive (FP):  {cm['FP']:<10} (Over-detected/noise)")
    print(f"  False Negative (FN):  {cm['FN']:<10} (Missed detections)")
    print(f"  True Negative (TN):   {cm['TN']:<10} (N/A for detection)")
    
    # Confusion Matrix - Primary
    print("\n📊 PRIMARY LIGATURES:")
    print("-"*50)
    p = results['primary']
    print(f"  TP: {p['TP']}  |  FP: {p['FP']}  |  FN: {p['FN']}")
    
    # Confusion Matrix - Diacritics
    print("\n📊 DIACRITICS:")
    print("-"*50)
    d = results['diacritics']
    print(f"  TP: {d['TP']}  |  FP: {d['FP']}  |  FN: {d['FN']}")
    
    # Accuracy Metrics
    print("\n" + "="*70)
    print("📈 ACCURACY METRICS:")
    print("="*70)
    m = results['metrics']
    print(f"\n  ✅ PRECISION:    {m['precision']*100:.2f}%")
    print(f"     Formula: TP/(TP+FP) = {cm['TP']}/({cm['TP']}+{cm['FP']})")
    
    print(f"\n  ✅ RECALL:       {m['recall']*100:.2f}%")
    print(f"     Formula: TP/(TP+FN) = {cm['TP']}/({cm['TP']}+{cm['FN']})")
    
    print(f"\n  ✅ F1-SCORE:     {m['f1_score']*100:.2f}%")
    print(f"     Formula: 2*(P*R)/(P+R)")
    
    print(f"\n  ✅ ACCURACY:     {m['accuracy']*100:.2f}%")
    print(f"     Formula: TP/Total Actual = {cm['TP']}/{results['actual']['total']}")
    
    print("\n" + "="*70)
    
    # Summary Box
    print("\n┌" + "─"*40 + "┐")
    print(f"│{'SUMMARY':^40}│")
    print("├" + "─"*40 + "┤")
    print(f"│  Precision:  {m['precision']*100:>6.2f}%                   │")
    print(f"│  Recall:     {m['recall']*100:>6.2f}%                   │")
    print(f"│  F1-Score:   {m['f1_score']*100:>6.2f}%                   │")
    print(f"│  Accuracy:   {m['accuracy']*100:>6.2f}%                   │")
    print("└" + "─"*40 + "┘")


def main():
    print("="*70)
    print("URDU OCR ACCURACY EVALUATION")
    print("="*70)
    
    if len(sys.argv) < 4:
        print("\nUsage:")
        print("  python quick_accuracy.py <image_path> <actual_primary> <actual_diacritics>")
        print("\nExample:")
        print("  python quick_accuracy.py input_images/4.png 650 400")
        print("\nNote:")
        print("  - actual_primary: Manually counted primary ligatures in the image")
        print("  - actual_diacritics: Manually counted dots/diacritics in the image")
        return
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"\n✗ Error: File not found: {image_path}")
        return
    
    try:
        actual_primary = int(sys.argv[2])
        actual_diacritics = int(sys.argv[3])
    except ValueError:
        print("\n✗ Error: actual_primary and actual_diacritics must be numbers")
        return
    
    print(f"\nImage: {image_path}")
    print(f"Actual Primary: {actual_primary}")
    print(f"Actual Diacritics: {actual_diacritics}")
    
    # Detect
    print("\nDetecting characters...")
    detected_primary, detected_diacritics = detect_and_separate(image_path)
    
    if detected_primary is None:
        return
    
    print(f"Detected Primary: {detected_primary}")
    print(f"Detected Diacritics: {detected_diacritics}")
    
    # Calculate accuracy
    results = calculate_accuracy(
        actual_primary, actual_diacritics,
        detected_primary, detected_diacritics
    )
    
    # Print results
    image_name = os.path.basename(image_path)
    print_results(results, image_name)
    
    # Save results
    output_folder = "evaluation_results"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    base_name = os.path.splitext(image_name)[0]
    results_path = os.path.join(output_folder, f"{base_name}_accuracy.json")
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_path}")


if __name__ == "__main__":
    main()
