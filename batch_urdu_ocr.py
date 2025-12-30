"""
Batch Urdu OCR System - Process Multiple Images
================================================
Based on paper: "Optical Character Recognition System for Urdu Words in Nastaliq Font"
By Safia Shabbir and Imran Siddiqi, Bahria University

Ek folder se sari images uthata hai aur:
1. Characters detect karta hai
2. Primary ligatures aur diacritics alag karta hai
3. Binary images alag folder mein save karta hai
4. Visualization images alag folder mein save karta hai
5. Har image ke characters separate folders mein save karta hai
"""

import cv2
import numpy as np
import os
from datetime import datetime


class BatchUrduOCR:
    """
    Batch processing for Urdu character detection
    """
    
    def __init__(self, input_folder="input_images", output_folder="output"):
        self.input_folder = input_folder
        self.output_folder = output_folder
        
        # Output folders
        self.binary_folder = os.path.join(output_folder, "binary_images")
        self.visualization_folder = os.path.join(output_folder, "visualization_images")
        self.characters_folder = os.path.join(output_folder, "extracted_characters")
        
        # Create folders
        folders = [
            self.input_folder,
            self.output_folder,
            self.binary_folder,
            self.visualization_folder,
            self.characters_folder
        ]
        
        for folder in folders:
            if not os.path.exists(folder):
                os.makedirs(folder)
                print(f"✓ Created: {folder}")
        
        self.min_component_area = 10
        self.supported_formats = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    
    def preprocess_binarization(self, image):
        """Step 1: Binarization - Otsu's thresholding"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary
    
    def extract_components(self, binary_image):
        """Step 2: Connected Component Labeling"""
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
            cx, cy = centroids[i]
            
            if area >= self.min_component_area:
                component_mask = (labels == i).astype(np.uint8) * 255
                component_image = component_mask[y:y+h, x:x+w]
                
                components.append({
                    'id': i, 'x': x, 'y': y, 'width': w, 'height': h,
                    'area': area, 'centroid_x': cx, 'centroid_y': cy,
                    'image': component_image, 'bbox': (x, y, w, h)
                })
        
        return components
    
    def detect_baseline(self, binary_image):
        """Step 3: Baseline Detection"""
        horizontal_projection = np.sum(binary_image, axis=1)
        baseline_y = np.argmax(horizontal_projection)
        return baseline_y
    
    def separate_primary_secondary(self, components, baseline_y):
        """Step 4: Separate Primary and Secondary Components"""
        primary_components = []
        secondary_components = []
        
        if not components:
            return [], []
        
        avg_height = np.mean([c['height'] for c in components])
        avg_area = np.mean([c['area'] for c in components])
        
        for comp in components:
            y_top = comp['y']
            y_bottom = comp['y'] + comp['height']
            touches_baseline = y_top <= baseline_y <= y_bottom
            is_small = comp['area'] < avg_area * 0.3 or comp['height'] < avg_height * 0.5
            
            if touches_baseline or (not is_small and comp['height'] > avg_height * 0.4):
                comp['type'] = 'primary'
                comp['position'] = 'baseline'
                primary_components.append(comp)
            else:
                comp['type'] = 'secondary'
                comp['position'] = 'above' if comp['centroid_y'] < baseline_y else 'below'
                secondary_components.append(comp)
        
        return primary_components, secondary_components
    
    def create_visualization(self, original_image, components, baseline_y):
        """Create visualization with bounding boxes"""
        if len(original_image.shape) == 2:
            vis_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        else:
            vis_image = original_image.copy()
        
        # Baseline
        cv2.line(vis_image, (0, baseline_y), (vis_image.shape[1], baseline_y), (255, 0, 0), 2)
        cv2.putText(vis_image, "Baseline", (10, baseline_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Bounding boxes
        for idx, comp in enumerate(components):
            x, y, w, h = comp['bbox']
            
            if comp['type'] == 'primary':
                color = (0, 255, 0)  # Green
                label = f"P{idx+1}"
            else:
                color = (0, 165, 255) if comp['position'] == 'above' else (0, 255, 255)
                label = f"D{idx+1}"
            
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(vis_image, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return vis_image
    
    def process_single_image(self, image_path, image_name):
        """Process one image"""
        print(f"\n{'='*60}")
        print(f"Processing: {image_name}")
        print('='*60)
        
        # Load
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"  ✗ ERROR: Could not load image!")
            return None
        
        print(f"  Size: {original_image.shape[1]}x{original_image.shape[0]}")
        
        base_name = os.path.splitext(image_name)[0]
        
        # Step 1: Binarization
        binary_image = self.preprocess_binarization(original_image)
        binary_path = os.path.join(self.binary_folder, f"{base_name}_binary.png")
        cv2.imwrite(binary_path, binary_image)
        print(f"  ✓ Binary saved: {base_name}_binary.png")
        
        # Step 2: Extract components
        components = self.extract_components(binary_image)
        print(f"  ✓ Components found: {len(components)}")
        
        if len(components) == 0:
            print("  ✗ No components detected!")
            return None
        
        # Step 3: Baseline
        baseline_y = self.detect_baseline(binary_image)
        
        # Step 4: Separate
        primary, secondary = self.separate_primary_secondary(components, baseline_y)
        print(f"  ✓ Primary ligatures: {len(primary)}")
        print(f"  ✓ Diacritics: {len(secondary)}")
        
        all_components = primary + secondary
        
        # Visualization
        visualization = self.create_visualization(original_image, all_components, baseline_y)
        viz_path = os.path.join(self.visualization_folder, f"{base_name}_visualization.png")
        cv2.imwrite(viz_path, visualization)
        print(f"  ✓ Visualization saved: {base_name}_visualization.png")
        
        # Save characters
        image_chars_folder = os.path.join(self.characters_folder, base_name)
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
        
        print(f"  ✓ Characters saved to: {base_name}/")
        
        return {
            'image_name': image_name,
            'primary_count': len(primary),
            'secondary_count': len(secondary),
            'total_components': len(all_components)
        }
    
    def process_all_images(self):
        """Process all images"""
        print("\n" + "="*60)
        print("BATCH URDU OCR SYSTEM")
        print("="*60)
        
        # Find images
        image_files = []
        for file in os.listdir(self.input_folder):
            ext = os.path.splitext(file)[1].lower()
            if ext in self.supported_formats:
                image_files.append(file)
        
        if len(image_files) == 0:
            print(f"\n✗ No images found in '{self.input_folder}'!")
            print(f"  Please add images (.png, .jpg, etc.) to the folder.")
            return
        
        print(f"\n✓ Found {len(image_files)} images")
        print(f"  Input: {self.input_folder}")
        print(f"  Output: {self.output_folder}")
        
        # Process each
        results = []
        for image_file in image_files:
            image_path = os.path.join(self.input_folder, image_file)
            result = self.process_single_image(image_path, image_file)
            if result:
                results.append(result)
        
        # Summary
        print("\n" + "="*60)
        print("PROCESSING COMPLETE!")
        print("="*60)
        print(f"\n✓ Images processed: {len(results)}/{len(image_files)}")
        
        if results:
            total_primary = sum(r['primary_count'] for r in results)
            total_secondary = sum(r['secondary_count'] for r in results)
            total_components = sum(r['total_components'] for r in results)
            
            print(f"  Primary ligatures: {total_primary}")
            print(f"  Diacritics: {total_secondary}")
            print(f"  Total components: {total_components}")
        
        print(f"\n📁 Output Structure:")
        print(f"   {self.output_folder}/")
        print(f"   ├── binary_images/")
        print(f"   ├── visualization_images/")
        print(f"   └── extracted_characters/")
        print(f"       ├── image_name/")
        print(f"       │   ├── primary_ligatures/")
        print(f"       │   └── diacritics/")
        
        return results


def main():
    import sys
    
    print("="*60)
    print("BATCH URDU CHARACTER DETECTION")
    print("Based on: Shabbir & Siddiqi, 2016")
    print("="*60)
    
    input_folder = "input_images"
    output_folder = "output"
    
    if len(sys.argv) > 1:
        input_folder = sys.argv[1]
    if len(sys.argv) > 2:
        output_folder = sys.argv[2]
    
    print(f"\nUsage: python batch_urdu_ocr.py [input_folder] [output_folder]")
    
    processor = BatchUrduOCR(input_folder=input_folder, output_folder=output_folder)
    processor.process_all_images()


if __name__ == "__main__":
    main()
