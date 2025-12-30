"""
Urdu OCR Accuracy Evaluation System
====================================
Calculate:
- True Positive (TP): Correctly detected characters
- False Positive (FP): Incorrectly detected (noise/wrong detection)
- False Negative (FN): Missed characters (not detected)
- True Negative (TN): Correctly rejected non-characters

Metrics:
- Accuracy = (TP + TN) / (TP + TN + FP + FN)
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
"""

import cv2
import numpy as np
import os
import json
from datetime import datetime


class AccuracyEvaluator:
    """
    Evaluate character detection accuracy
    """
    
    def __init__(self, output_folder="evaluation_results"):
        self.output_folder = output_folder
        self.ground_truth_folder = os.path.join(output_folder, "ground_truth")
        self.comparison_folder = os.path.join(output_folder, "comparisons")
        
        for folder in [self.output_folder, self.ground_truth_folder, self.comparison_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)
        
        self.min_component_area = 10
        self.iou_threshold = 0.5  # IoU threshold for matching
    
    def preprocess_binarization(self, image):
        """Binarization using Otsu's thresholding"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary
    
    def detect_components(self, binary_image):
        """Detect components using connected component labeling"""
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_image, connectivity=8
        )
        
        components = []
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            
            if area >= self.min_component_area:
                components.append({
                    'x': x, 'y': y, 'width': w, 'height': h,
                    'area': area, 'bbox': (x, y, w, h)
                })
        
        return components
    
    def calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union (IoU) between two boxes
        box format: (x, y, w, h)
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def create_ground_truth_interactive(self, image_path):
        """
        Interactive tool to create ground truth by clicking on characters
        Press 'p' for primary, 'd' for diacritic, 'u' to undo, 's' to save, 'q' to quit
        """
        print("\n" + "="*60)
        print("GROUND TRUTH ANNOTATION TOOL")
        print("="*60)
        print("Instructions:")
        print("  - Click to mark character location")
        print("  - Press 'p' then click: Mark as Primary ligature")
        print("  - Press 'd' then click: Mark as Diacritic")
        print("  - Press 'u': Undo last annotation")
        print("  - Press 's': Save ground truth")
        print("  - Press 'q': Quit")
        print("="*60)
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load {image_path}")
            return None
        
        display_image = image.copy()
        annotations = []
        current_type = 'primary'  # Default type
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal annotations, display_image, current_type
            
            if event == cv2.EVENT_LBUTTONDOWN:
                # Add annotation at click position
                # Estimate bounding box around click
                box_size = 30
                annotation = {
                    'x': max(0, x - box_size//2),
                    'y': max(0, y - box_size//2),
                    'width': box_size,
                    'height': box_size,
                    'type': current_type,
                    'center': (x, y)
                }
                annotations.append(annotation)
                
                # Draw on display
                color = (0, 255, 0) if current_type == 'primary' else (0, 165, 255)
                cv2.circle(display_image, (x, y), 5, color, -1)
                cv2.rectangle(display_image, 
                            (annotation['x'], annotation['y']),
                            (annotation['x'] + annotation['width'], 
                             annotation['y'] + annotation['height']),
                            color, 2)
                
                print(f"  Added {current_type} at ({x}, {y})")
        
        cv2.namedWindow("Ground Truth Annotation")
        cv2.setMouseCallback("Ground Truth Annotation", mouse_callback)
        
        while True:
            cv2.imshow("Ground Truth Annotation", display_image)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('p'):
                current_type = 'primary'
                print("  Mode: PRIMARY ligature")
            elif key == ord('d'):
                current_type = 'diacritic'
                print("  Mode: DIACRITIC")
            elif key == ord('u') and annotations:
                annotations.pop()
                display_image = image.copy()
                for ann in annotations:
                    color = (0, 255, 0) if ann['type'] == 'primary' else (0, 165, 255)
                    cv2.circle(display_image, ann['center'], 5, color, -1)
                    cv2.rectangle(display_image,
                                (ann['x'], ann['y']),
                                (ann['x'] + ann['width'], ann['y'] + ann['height']),
                                color, 2)
                print("  Undo last annotation")
            elif key == ord('s'):
                # Save ground truth
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                gt_path = os.path.join(self.ground_truth_folder, f"{base_name}_gt.json")
                
                gt_data = {
                    'image': image_path,
                    'annotations': annotations,
                    'total_primary': sum(1 for a in annotations if a['type'] == 'primary'),
                    'total_diacritics': sum(1 for a in annotations if a['type'] == 'diacritic'),
                    'total': len(annotations)
                }
                
                with open(gt_path, 'w') as f:
                    json.dump(gt_data, f, indent=2)
                
                print(f"\n✓ Ground truth saved: {gt_path}")
                print(f"  Total annotations: {len(annotations)}")
                break
            elif key == ord('q'):
                print("  Cancelled")
                break
        
        cv2.destroyAllWindows()
        return annotations
    
    def create_ground_truth_from_count(self, image_path, actual_primary, actual_diacritics):
        """
        Simple method: Just provide the actual counts
        User manually counts the actual characters in the image
        """
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        gt_path = os.path.join(self.ground_truth_folder, f"{base_name}_gt.json")
        
        gt_data = {
            'image': image_path,
            'actual_primary': actual_primary,
            'actual_diacritics': actual_diacritics,
            'actual_total': actual_primary + actual_diacritics,
            'method': 'count_based'
        }
        
        with open(gt_path, 'w') as f:
            json.dump(gt_data, f, indent=2)
        
        print(f"✓ Ground truth saved: {gt_path}")
        return gt_data
    
    def evaluate_detection(self, image_path, ground_truth=None):
        """
        Evaluate detection accuracy for an image
        """
        print("\n" + "="*60)
        print(f"EVALUATING: {os.path.basename(image_path)}")
        print("="*60)
        
        # Load image and detect
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image")
            return None
        
        binary = self.preprocess_binarization(image)
        detected = self.detect_components(binary)
        
        # Separate primary and diacritics
        baseline_y = np.argmax(np.sum(binary, axis=1))
        
        detected_primary = []
        detected_diacritics = []
        
        if detected:
            avg_height = np.mean([c['height'] for c in detected])
            avg_area = np.mean([c['area'] for c in detected])
            
            for comp in detected:
                y_top = comp['y']
                y_bottom = comp['y'] + comp['height']
                touches_baseline = y_top <= baseline_y <= y_bottom
                is_small = comp['area'] < avg_area * 0.3 or comp['height'] < avg_height * 0.5
                
                if touches_baseline or (not is_small and comp['height'] > avg_height * 0.4):
                    detected_primary.append(comp)
                else:
                    detected_diacritics.append(comp)
        
        # Load ground truth
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        gt_path = os.path.join(self.ground_truth_folder, f"{base_name}_gt.json")
        
        if ground_truth is None and os.path.exists(gt_path):
            with open(gt_path, 'r') as f:
                ground_truth = json.load(f)
        
        if ground_truth is None:
            print("  ✗ No ground truth found!")
            print(f"  Detected: {len(detected_primary)} primary, {len(detected_diacritics)} diacritics")
            return None
        
        # Calculate metrics
        results = self.calculate_metrics(
            detected_primary, detected_diacritics,
            ground_truth, image, base_name
        )
        
        return results
    
    def calculate_metrics(self, detected_primary, detected_diacritics, 
                         ground_truth, image, image_name):
        """
        Calculate TP, FP, FN, TN and accuracy metrics
        """
        # Get actual counts from ground truth
        if 'actual_primary' in ground_truth:
            # Count-based ground truth
            actual_primary = ground_truth['actual_primary']
            actual_diacritics = ground_truth['actual_diacritics']
            actual_total = ground_truth['actual_total']
        else:
            # Annotation-based ground truth
            actual_primary = ground_truth.get('total_primary', 0)
            actual_diacritics = ground_truth.get('total_diacritics', 0)
            actual_total = ground_truth.get('total', 0)
        
        detected_primary_count = len(detected_primary)
        detected_diacritics_count = len(detected_diacritics)
        detected_total = detected_primary_count + detected_diacritics_count
        
        # Calculate metrics for PRIMARY ligatures
        print("\n📊 PRIMARY LIGATURES:")
        print("-" * 40)
        
        # TP: Correctly detected (min of actual and detected, assuming detection is mostly correct)
        # This is a simplified approach - for exact TP you need bbox matching
        tp_primary = min(detected_primary_count, actual_primary)
        
        # FP: Detected but shouldn't be (over-detection)
        fp_primary = max(0, detected_primary_count - actual_primary)
        
        # FN: Should be detected but wasn't (under-detection)
        fn_primary = max(0, actual_primary - detected_primary_count)
        
        print(f"  Actual count:    {actual_primary}")
        print(f"  Detected count:  {detected_primary_count}")
        print(f"  True Positive:   {tp_primary}")
        print(f"  False Positive:  {fp_primary}")
        print(f"  False Negative:  {fn_primary}")
        
        # Calculate metrics for DIACRITICS
        print("\n📊 DIACRITICS:")
        print("-" * 40)
        
        tp_diacritics = min(detected_diacritics_count, actual_diacritics)
        fp_diacritics = max(0, detected_diacritics_count - actual_diacritics)
        fn_diacritics = max(0, actual_diacritics - detected_diacritics_count)
        
        print(f"  Actual count:    {actual_diacritics}")
        print(f"  Detected count:  {detected_diacritics_count}")
        print(f"  True Positive:   {tp_diacritics}")
        print(f"  False Positive:  {fp_diacritics}")
        print(f"  False Negative:  {fn_diacritics}")
        
        # OVERALL metrics
        print("\n📊 OVERALL RESULTS:")
        print("-" * 40)
        
        tp_total = tp_primary + tp_diacritics
        fp_total = fp_primary + fp_diacritics
        fn_total = fn_primary + fn_diacritics
        
        # For character detection, TN is typically 0 or not applicable
        # But we can estimate it as the correctly rejected non-character regions
        tn_total = 0  # Not easily calculable without full image annotation
        
        print(f"  Actual Total:      {actual_total}")
        print(f"  Detected Total:    {detected_total}")
        print(f"  True Positive:     {tp_total}")
        print(f"  False Positive:    {fp_total}")
        print(f"  False Negative:    {fn_total}")
        print(f"  True Negative:     {tn_total} (N/A for detection)")
        
        # Calculate Accuracy Metrics
        print("\n" + "="*60)
        print("📈 ACCURACY METRICS:")
        print("="*60)
        
        # Precision = TP / (TP + FP)
        precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
        
        # Recall = TP / (TP + FN)
        recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
        
        # F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Accuracy = TP / Actual Total (Detection Rate)
        accuracy = tp_total / actual_total if actual_total > 0 else 0
        
        # Detection Rate
        detection_rate = detected_total / actual_total if actual_total > 0 else 0
        
        print(f"\n  ✅ Precision:       {precision*100:.2f}%")
        print(f"  ✅ Recall:          {recall*100:.2f}%")
        print(f"  ✅ F1-Score:        {f1_score*100:.2f}%")
        print(f"  ✅ Accuracy:        {accuracy*100:.2f}%")
        print(f"  📊 Detection Rate:  {detection_rate*100:.2f}%")
        
        # Primary ligatures metrics
        precision_primary = tp_primary / detected_primary_count if detected_primary_count > 0 else 0
        recall_primary = tp_primary / actual_primary if actual_primary > 0 else 0
        f1_primary = 2 * (precision_primary * recall_primary) / (precision_primary + recall_primary) if (precision_primary + recall_primary) > 0 else 0
        
        print(f"\n  Primary Ligatures:")
        print(f"    Precision: {precision_primary*100:.2f}%")
        print(f"    Recall:    {recall_primary*100:.2f}%")
        print(f"    F1-Score:  {f1_primary*100:.2f}%")
        
        # Diacritics metrics
        precision_diacritics = tp_diacritics / detected_diacritics_count if detected_diacritics_count > 0 else 0
        recall_diacritics = tp_diacritics / actual_diacritics if actual_diacritics > 0 else 0
        f1_diacritics = 2 * (precision_diacritics * recall_diacritics) / (precision_diacritics + recall_diacritics) if (precision_diacritics + recall_diacritics) > 0 else 0
        
        print(f"\n  Diacritics:")
        print(f"    Precision: {precision_diacritics*100:.2f}%")
        print(f"    Recall:    {recall_diacritics*100:.2f}%")
        print(f"    F1-Score:  {f1_diacritics*100:.2f}%")
        
        # Save results
        results = {
            'image_name': image_name,
            'timestamp': datetime.now().isoformat(),
            'actual': {
                'primary': actual_primary,
                'diacritics': actual_diacritics,
                'total': actual_total
            },
            'detected': {
                'primary': detected_primary_count,
                'diacritics': detected_diacritics_count,
                'total': detected_total
            },
            'confusion_matrix': {
                'TP': tp_total,
                'FP': fp_total,
                'FN': fn_total,
                'TN': tn_total
            },
            'metrics': {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'accuracy': accuracy,
                'detection_rate': detection_rate
            },
            'primary_metrics': {
                'precision': precision_primary,
                'recall': recall_primary,
                'f1_score': f1_primary
            },
            'diacritics_metrics': {
                'precision': precision_diacritics,
                'recall': recall_diacritics,
                'f1_score': f1_diacritics
            }
        }
        
        # Save to JSON
        results_path = os.path.join(self.output_folder, f"{image_name}_evaluation.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved: {results_path}")
        
        return results
    
    def print_summary_table(self, results):
        """Print a nice summary table"""
        print("\n" + "="*70)
        print("EVALUATION SUMMARY TABLE")
        print("="*70)
        print(f"{'Metric':<25} {'Value':<15} {'Percentage':<15}")
        print("-"*70)
        
        cm = results['confusion_matrix']
        m = results['metrics']
        
        print(f"{'True Positive (TP)':<25} {cm['TP']:<15}")
        print(f"{'False Positive (FP)':<25} {cm['FP']:<15}")
        print(f"{'False Negative (FN)':<25} {cm['FN']:<15}")
        print(f"{'True Negative (TN)':<25} {cm['TN']:<15} {'N/A':<15}")
        print("-"*70)
        print(f"{'Precision':<25} {'':<15} {m['precision']*100:.2f}%")
        print(f"{'Recall':<25} {'':<15} {m['recall']*100:.2f}%")
        print(f"{'F1-Score':<25} {'':<15} {m['f1_score']*100:.2f}%")
        print(f"{'Accuracy':<25} {'':<15} {m['accuracy']*100:.2f}%")
        print("="*70)


def main():
    import sys
    
    print("="*60)
    print("URDU OCR ACCURACY EVALUATION SYSTEM")
    print("="*60)
    print("\nOptions:")
    print("  1. Set ground truth for an image")
    print("  2. Evaluate detection accuracy")
    print("  3. Evaluate with manual count input")
    print("="*60)
    
    evaluator = AccuracyEvaluator()
    
    while True:
        print("\nEnter choice (1/2/3) or 'q' to quit: ", end="")
        choice = input().strip()
        
        if choice == 'q':
            break
        elif choice == '1':
            print("Enter image path: ", end="")
            image_path = input().strip()
            if os.path.exists(image_path):
                try:
                    evaluator.create_ground_truth_interactive(image_path)
                except Exception as e:
                    print(f"Error: {e}")
                    print("Use option 3 for manual count input instead.")
            else:
                print(f"File not found: {image_path}")
        
        elif choice == '2':
            print("Enter image path: ", end="")
            image_path = input().strip()
            if os.path.exists(image_path):
                results = evaluator.evaluate_detection(image_path)
                if results:
                    evaluator.print_summary_table(results)
            else:
                print(f"File not found: {image_path}")
        
        elif choice == '3':
            print("Enter image path: ", end="")
            image_path = input().strip()
            
            if not os.path.exists(image_path):
                print(f"File not found: {image_path}")
                continue
            
            print("Enter ACTUAL primary ligature count: ", end="")
            try:
                actual_primary = int(input().strip())
            except:
                print("Invalid number")
                continue
            
            print("Enter ACTUAL diacritics count: ", end="")
            try:
                actual_diacritics = int(input().strip())
            except:
                print("Invalid number")
                continue
            
            # Save ground truth
            gt = evaluator.create_ground_truth_from_count(
                image_path, actual_primary, actual_diacritics
            )
            
            # Evaluate
            results = evaluator.evaluate_detection(image_path, gt)
            if results:
                evaluator.print_summary_table(results)
        
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()
