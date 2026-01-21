"""
Accuracy Calculator for Complete OCR System
============================================
Results folder se directly accuracy calculate karta hai

Usage: python calculate_accuracy.py <results_folder> <actual_ligatures> <actual_orphans>
"""

import cv2
import numpy as np
import os
import sys
import json


def load_results(results_folder):
    """Load results from JSON file"""
    json_path = os.path.join(results_folder, 'results.json')
    
    if not os.path.exists(json_path):
        print(f"❌ Error: results.json not found in {results_folder}")
        return None
    
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_metrics(actual_ligatures, actual_orphans, detected_ligatures, detected_orphans):
    """Calculate accuracy metrics"""
    # Ligatures
    tp_lig = min(actual_ligatures, detected_ligatures)
    fp_lig = max(0, detected_ligatures - actual_ligatures)
    fn_lig = max(0, actual_ligatures - detected_ligatures)
    
    # Orphans
    tp_orph = min(actual_orphans, detected_orphans)
    fp_orph = max(0, detected_orphans - actual_orphans)
    fn_orph = max(0, actual_orphans - detected_orphans)
    
    # Total
    tp_total = tp_lig + tp_orph
    fp_total = fp_lig + fp_orph
    fn_total = fn_lig + fn_orph
    
    # Metrics
    precision = (tp_total / (tp_total + fp_total) * 100) if (tp_total + fp_total) > 0 else 0
    recall = (tp_total / (tp_total + fn_total) * 100) if (tp_total + fn_total) > 0 else 0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    
    actual_total = actual_ligatures + actual_orphans
    accuracy = (tp_total / actual_total * 100) if actual_total > 0 else 0
    
    return {
        'actual_ligatures': actual_ligatures,
        'actual_orphans': actual_orphans,
        'actual_total': actual_total,
        'detected_ligatures': detected_ligatures,
        'detected_orphans': detected_orphans,
        'detected_total': detected_ligatures + detected_orphans,
        'tp_ligatures': tp_lig,
        'fp_ligatures': fp_lig,
        'fn_ligatures': fn_lig,
        'tp_orphans': tp_orph,
        'fp_orphans': fp_orph,
        'fn_orphans': fn_orph,
        'tp_total': tp_total,
        'fp_total': fp_total,
        'fn_total': fn_total,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy
    }


def print_report(results_data, metrics):
    """Print detailed accuracy report"""
    print("\n" + "="*70)
    print(f"📊 ACCURACY REPORT: {results_data['image']}")
    print("="*70)
    
    # Detection info
    print(f"\n📋 DETECTION INFO:")
    print(f"  Lines detected: {results_data['total_lines']}")
    print(f"  Total components: {results_data['summary']['total_components']}")
    
    # Per-line breakdown
    if len(results_data['per_line_results']) > 1:
        print(f"\n  Per-line breakdown:")
        for line_result in results_data['per_line_results']:
            print(f"    Line {line_result['line']}: {line_result['ligatures']} ligatures, "
                  f"{line_result['orphan_diacritics']} orphans")
    
    # Counts comparison
    print("\n" + "="*70)
    print("📈 COUNTS COMPARISON:")
    print("-"*70)
    print(f"{'Category':<25} {'Actual':<15} {'Detected':<15} {'Match'}")
    print("-"*70)
    
    lig_match = "✓" if metrics['actual_ligatures'] == metrics['detected_ligatures'] else "✗"
    orph_match = "✓" if metrics['actual_orphans'] == metrics['detected_orphans'] else "✗"
    
    print(f"{'Ligatures':<25} {metrics['actual_ligatures']:<15} {metrics['detected_ligatures']:<15} {lig_match}")
    print(f"{'Orphan Diacritics':<25} {metrics['actual_orphans']:<15} {metrics['detected_orphans']:<15} {orph_match}")
    print(f"{'TOTAL':<25} {metrics['actual_total']:<15} {metrics['detected_total']:<15}")
    
    # Confusion Matrix
    print("\n" + "="*70)
    print("🔍 CONFUSION MATRIX:")
    print("="*70)
    
    print("\n📊 Ligatures:")
    print(f"  ✅ True Positive (TP):   {metrics['tp_ligatures']}")
    print(f"  ❌ False Positive (FP):  {metrics['fp_ligatures']}")
    print(f"  ⚠️  False Negative (FN):  {metrics['fn_ligatures']}")
    
    print("\n📊 Orphan Diacritics:")
    print(f"  ✅ True Positive (TP):   {metrics['tp_orphans']}")
    print(f"  ❌ False Positive (FP):  {metrics['fp_orphans']}")
    print(f"  ⚠️  False Negative (FN):  {metrics['fn_orphans']}")
    
    print("\n📊 OVERALL:")
    print(f"  ✅ True Positive (TP):   {metrics['tp_total']:<10} (Correctly detected)")
    print(f"  ❌ False Positive (FP):  {metrics['fp_total']:<10} (Over-detected)")
    print(f"  ⚠️  False Negative (FN):  {metrics['fn_total']:<10} (Missed)")
    
    # Accuracy Metrics
    print("\n" + "="*70)
    print("📈 ACCURACY METRICS:")
    print("="*70)
    
    print(f"\n  🎯 PRECISION: {metrics['precision']:.2f}%")
    print(f"     What % of detected are correct")
    print(f"     Formula: TP/(TP+FP) = {metrics['tp_total']}/({metrics['tp_total']}+{metrics['fp_total']})")
    
    print(f"\n  🎯 RECALL: {metrics['recall']:.2f}%")
    print(f"     What % of actual were detected")
    print(f"     Formula: TP/(TP+FN) = {metrics['tp_total']}/({metrics['tp_total']}+{metrics['fn_total']})")
    
    print(f"\n  🎯 F1-SCORE: {metrics['f1_score']:.2f}%")
    print(f"     Balance between Precision and Recall")
    
    print(f"\n  🎯 ACCURACY: {metrics['accuracy']:.2f}%")
    print(f"     Overall correctness")
    print(f"     Formula: TP/Total = {metrics['tp_total']}/{metrics['actual_total']}")
    
    # Final box
    print("\n" + "="*70)
    print("✅ FINAL RESULT:")
    print("="*70)
    print("┌─────────────────────────────────────────┐")
    print(f"│ Precision:  {metrics['precision']:>7.2f}%                │")
    print(f"│ Recall:     {metrics['recall']:>7.2f}%                │")
    print(f"│ F1-Score:   {metrics['f1_score']:>7.2f}%                │")
    print(f"│ Accuracy:   {metrics['accuracy']:>7.2f}%                │")
    print("└─────────────────────────────────────────┘")


def main():
    print("="*70)
    print("ACCURACY CALCULATOR")
    print("="*70)
    
    if len(sys.argv) < 4:
        print("\n📝 Usage:")
        print("   python calculate_accuracy.py <results_folder> <actual_ligatures> <actual_orphans>")
        print("\n📌 Example:")
        print('   python calculate_accuracy.py "results_img3" 10 2')
        print("\n❓ Parameters:")
        print("   results_folder: OCR results folder name (e.g., results_img3)")
        print("   actual_ligatures: Actual number of ligatures in image")
        print("   actual_orphans: Actual number of orphan diacritics")
        print("\n💡 Tip: results_folder automatically created by complete_urdu_ocr.py")
        return
    
    results_folder = sys.argv[1]
    
    try:
        actual_ligatures = int(sys.argv[2])
        actual_orphans = int(sys.argv[3])
    except ValueError:
        print("❌ Error: Please provide numbers for actual counts")
        return
    
    if not os.path.exists(results_folder):
        print(f"❌ Error: Folder not found: {results_folder}")
        return
    
    # Load results
    print(f"\n📂 Loading results from: {results_folder}")
    results_data = load_results(results_folder)
    
    if results_data is None:
        return
    
    detected_ligatures = results_data['summary']['total_ligatures']
    detected_orphans = results_data['summary']['orphan_diacritics']
    
    print(f"\n✓ Loaded results:")
    print(f"  Detected ligatures: {detected_ligatures}")
    print(f"  Detected orphan diacritics: {detected_orphans}")
    
    # Calculate metrics
    print("\n🧮 Calculating accuracy metrics...")
    metrics = calculate_metrics(
        actual_ligatures, actual_orphans,
        detected_ligatures, detected_orphans
    )
    
    # Print report
    print_report(results_data, metrics)
    
    # Save accuracy results
    accuracy_file = os.path.join(results_folder, 'accuracy.json')
    with open(accuracy_file, 'w', encoding='utf-8') as f:
        json.dump({
            'actual': {
                'ligatures': actual_ligatures,
                'orphan_diacritics': actual_orphans,
                'total': actual_ligatures + actual_orphans
            },
            'detected': {
                'ligatures': detected_ligatures,
                'orphan_diacritics': detected_orphans,
                'total': detected_ligatures + detected_orphans
            },
            'metrics': metrics
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Accuracy saved to: {accuracy_file}")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
