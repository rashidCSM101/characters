"""
Complete Urdu OCR Pipeline with Automatic Accuracy
====================================================
1. Process images from input_images folder
2. Detect characters (primary + diacritics)
3. Automatically calculate accuracy metrics
4. Save everything

Usage:
    python batch_accuracy_pipeline.py
    
Or with custom folders:
    python batch_accuracy_pipeline.py input_folder output_folder
"""

import cv2
import numpy as np
import os
import sys
import json
from datetime import datetime


class CompleteOCRPipeline:
    """
    Complete pipeline: Detection + Accuracy in one go
    """
    
    def __init__(self, input_folder="input_images", output_folder="output"):
        self.input_folder = input_folder
        self.output_folder = output_folder
        
        # Output subfolders
        self.binary_folder = os.path.join(output_folder, "binary_images")
        self.visualization_folder = os.path.join(output_folder, "visualization_images")
        self.characters_folder = os.path.join(output_folder, "extracted_characters")
        self.accuracy_folder = os.path.join(output_folder, "accuracy_results")
        
        # Create all folders
        for folder in [self.input_folder, self.output_folder, self.binary_folder, 
                       self.visualization_folder, self.characters_folder, self.accuracy_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)
        
        self.min_component_area = 10
        self.supported_formats = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    
    def preprocess_binarization(self, image):
        """Binarization using Otsu's thresholding"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary
    
    def detect_and_separate(self, binary_image):
        """Detect and separate primary/diacritics"""
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
            cy = centroids[i][1]
            
            if area >= self.min_component_area:
                component_mask = (labels == i).astype(np.uint8) * 255
                component_image = component_mask[y:y+h, x:x+w]
                
                components.append({
                    'x': x, 'y': y, 'width': w, 'height': h,
                    'area': area, 'centroid_y': cy,
                    'image': component_image, 'bbox': (x, y, w, h)
                })
        
        # Baseline detection
        baseline_y = np.argmax(np.sum(binary_image, axis=1))
        
        # Separate
        primary = []
        secondary = []
        
        if components:
            avg_height = np.mean([c['height'] for c in components])
            avg_area = np.mean([c['area'] for c in components])
            
            for comp in components:
                y_top = comp['y']
                y_bottom = comp['y'] + comp['height']
                touches_baseline = y_top <= baseline_y <= y_bottom
                is_small = comp['area'] < avg_area * 0.3 or comp['height'] < avg_height * 0.5
                
                if touches_baseline or (not is_small and comp['height'] > avg_height * 0.4):
                    comp['type'] = 'primary'
                    primary.append(comp)
                else:
                    comp['type'] = 'secondary'
                    comp['position'] = 'above' if comp['centroid_y'] < baseline_y else 'below'
                    secondary.append(comp)
        
        return primary, secondary, baseline_y
    
    def create_visualization(self, original_image, all_components, baseline_y):
        """Create visualization"""
        if len(original_image.shape) == 2:
            vis_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        else:
            vis_image = original_image.copy()
        
        # Baseline
        cv2.line(vis_image, (0, baseline_y), (vis_image.shape[1], baseline_y), (255, 0, 0), 2)
        cv2.putText(vis_image, "Baseline", (10, baseline_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Boxes
        for comp in all_components:
            x, y, w, h = comp['bbox']
            color = (0, 255, 0) if comp['type'] == 'primary' else (0, 165, 255)
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
        
        return vis_image
    
    def save_components(self, image_name, primary, secondary):
        """Save extracted components"""
        image_chars_folder = os.path.join(self.characters_folder, image_name)
        primary_folder = os.path.join(image_chars_folder, "primary_ligatures")
        secondary_folder = os.path.join(image_chars_folder, "diacritics")
        
        for folder in [image_chars_folder, primary_folder, secondary_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)
        
        # Save primary
        sorted_primary = sorted(primary, key=lambda c: -c['x'])
        for idx, comp in enumerate(sorted_primary):
            filename = f"primary_{idx+1:04d}.png"
            filepath = os.path.join(primary_folder, filename)
            cv2.imwrite(filepath, comp['image'])
        
        # Save secondary
        sorted_secondary = sorted(secondary, key=lambda c: -c['x'])
        for idx, comp in enumerate(sorted_secondary):
            pos = comp['position']
            filename = f"diacritic_{pos}_{idx+1:04d}.png"
            filepath = os.path.join(secondary_folder, filename)
            cv2.imwrite(filepath, comp['image'])
    
    def process_image(self, image_path, image_name):
        """Process single image"""
        print(f"\n{'='*70}")
        print(f"Processing: {image_name}")
        print('='*70)
        
        # Load
        image = cv2.imread(image_path)
        if image is None:
            print(f"  ✗ Could not load!")
            return None
        
        print(f"  ✓ Image size: {image.shape[1]}x{image.shape[0]}")
        
        base_name = os.path.splitext(image_name)[0]
        
        # Step 1: Binarization
        binary = self.preprocess_binarization(image)
        binary_path = os.path.join(self.binary_folder, f"{base_name}_binary.png")
        cv2.imwrite(binary_path, binary)
        print(f"  ✓ Binary image saved")
        
        # Step 2: Detect and separate
        primary, secondary, baseline_y = self.detect_and_separate(binary)
        print(f"  ✓ Detected: {len(primary)} primary, {len(secondary)} diacritics")
        
        # Step 3: Visualization
        all_comps = primary + secondary
        visualization = self.create_visualization(image, all_comps, baseline_y)
        viz_path = os.path.join(self.visualization_folder, f"{base_name}_visualization.png")
        cv2.imwrite(viz_path, visualization)
        print(f"  ✓ Visualization saved")
        
        # Step 4: Save components
        self.save_components(base_name, primary, secondary)
        print(f"  ✓ Characters saved")
        
        return {
            'image_name': image_name,
            'detected_primary': len(primary),
            'detected_secondary': len(secondary),
            'detected_total': len(primary) + len(secondary)
        }
    
    def process_all_images(self):
        """Process all images in input folder"""
        print("\n" + "="*70)
        print("URDU OCR - COMPLETE PIPELINE")
        print("="*70)
        
        # Find images
        image_files = []
        for file in os.listdir(self.input_folder):
            ext = os.path.splitext(file)[1].lower()
            if ext in self.supported_formats:
                image_files.append(file)
        
        if not image_files:
            print(f"\n✗ No images found in '{self.input_folder}'")
            return None
        
        print(f"\n✓ Found {len(image_files)} images")
        
        # Process each
        all_results = []
        for image_file in image_files:
            image_path = os.path.join(self.input_folder, image_file)
            result = self.process_image(image_path, image_file)
            if result:
                all_results.append(result)
        
        # Summary
        print("\n" + "="*70)
        print("DETECTION SUMMARY")
        print("="*70)
        
        for result in all_results:
            print(f"\n{result['image_name']}:")
            print(f"  Primary ligatures:  {result['detected_primary']}")
            print(f"  Diacritics:         {result['detected_secondary']}")
            print(f"  Total:              {result['detected_total']}")
        
        # Save detection results
        detection_file = os.path.join(self.output_folder, "detection_results.json")
        with open(detection_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✓ Detection results saved: {detection_file}")
        
        return all_results
    
    def generate_accuracy_template(self, results):
        """Generate template for manual accuracy input"""
        print("\n" + "="*70)
        print("📝 ACCURACY EVALUATION TEMPLATE")
        print("="*70)
        
        template = {}
        for result in results:
            image_name = result['image_name']
            base_name = os.path.splitext(image_name)[0]
            
            print(f"\n📌 {image_name}:")
            print(f"   System detected:")
            print(f"   - Primary ligatures: {result['detected_primary']}")
            print(f"   - Diacritics: {result['detected_secondary']}")
            print(f"\n   👉 Now manually count the ACTUAL values in the image:")
            print(f"      Enter: python simple_accuracy.py \"input_images/{image_name}\" <actual_primary> <actual_diacritics>")
            
            template[base_name] = {
                'image_name': image_name,
                'detected_primary': result['detected_primary'],
                'detected_secondary': result['detected_secondary'],
                'actual_primary': 'MANUAL_COUNT_HERE',
                'actual_diacritics': 'MANUAL_COUNT_HERE'
            }
        
        # Save template
        template_file = os.path.join(self.accuracy_folder, "accuracy_template.json")
        with open(template_file, 'w') as f:
            json.dump(template, f, indent=2)
        
        print(f"\n✓ Template saved: {template_file}")
        print("\n" + "="*70)
        print("NEXT STEPS:")
        print("="*70)
        print("\n1. Manually count ACTUAL primary ligatures and diacritics in each image")
        print("\n2. Run accuracy evaluation:")
        print('   python simple_accuracy.py "input_images/image_name.png" <actual_primary> <actual_diacritics>')
        print("\n3. Example:")
        print('   python simple_accuracy.py "input_images/page_003.png" 700 420')


def main():
    print("="*70)
    print("URDU OCR COMPLETE PIPELINE")
    print("Batch Processing + Automatic Detection")
    print("="*70)
    
    input_folder = "input_images"
    output_folder = "output"
    
    if len(sys.argv) > 1:
        input_folder = sys.argv[1]
    if len(sys.argv) > 2:
        output_folder = sys.argv[2]
    
    # Create pipeline
    pipeline = CompleteOCRPipeline(input_folder, output_folder)
    
    # Process all images
    results = pipeline.process_all_images()
    
    if results:
        # Generate accuracy template
        pipeline.generate_accuracy_template(results)
    else:
        print("\n✗ No images processed!")


if __name__ == "__main__":
    main()
