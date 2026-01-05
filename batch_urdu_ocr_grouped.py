"""
Batch Urdu OCR System - WITH DIACRITIC GROUPING
================================================
ENHANCEMENT: Nuqte (diacritics) ligatures ke saath grouped rahenge
"""

import cv2
import numpy as np
import os
from datetime import datetime


class BatchUrduOCRGrouped:
    """
    Enhanced: Diacritics ko unke parent ligatures ke saath group karta hai
    """
    
    def __init__(self, input_folder="input_images", output_folder="output_grouped"):
        self.input_folder = input_folder
        self.output_folder = output_folder
        
        # Output folders
        self.binary_folder = os.path.join(output_folder, "binary_images")
        self.visualization_folder = os.path.join(output_folder, "visualization_images")
        self.characters_folder = os.path.join(output_folder, "extracted_characters")
        self.preprocessing_folder = os.path.join(output_folder, "preprocessing_steps")
        
        # Create folders
        folders = [
            self.input_folder, self.output_folder,
            self.binary_folder, self.visualization_folder,
            self.characters_folder, self.preprocessing_folder
        ]
        
        for folder in folders:
            if not os.path.exists(folder):
                os.makedirs(folder)
                print(f"✓ Created: {folder}")
        
        self.min_component_area = 3
        self.supported_formats = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
        self.adaptive_block_size = 13
        self.adaptive_C = 3
    
    def enhance_contrast(self, gray_image):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray_image)
    
    def bilateral_denoise(self, gray_image):
        return cv2.bilateralFilter(gray_image, 9, 75, 75)
    
    def detect_skew_angle(self, binary_image):
        edges = cv2.Canny(binary_image, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                               minLineLength=100, maxLineGap=10)
        
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
    
    def correct_skew(self, image, angle):
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
        
        return cv2.warpAffine(image, M, (new_w, new_h), 
                             flags=cv2.INTER_CUBIC, 
                             borderMode=cv2.BORDER_REPLICATE)
    
    def preprocess_image(self, image):
        """Enhanced preprocessing pipeline"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        denoised = self.bilateral_denoise(gray)
        enhanced = self.enhance_contrast(denoised)
        
        _, initial_binary = cv2.threshold(enhanced, 0, 255, 
                                         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        skew_angle = self.detect_skew_angle(initial_binary)
        if abs(skew_angle) > 0.5:
            enhanced = self.correct_skew(enhanced, skew_angle)
            print(f"  ↳ Skew corrected: {skew_angle:.2f}°")
        
        binary = cv2.adaptiveThreshold(enhanced, 255, 
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 
                                       self.adaptive_block_size, self.adaptive_C)
        
        kernel_tiny = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_tiny)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_small)
        
        return binary
    
    def group_diacritics_with_ligatures(self, primary_components, secondary_components):
        """
        🎯 KEY FUNCTION: Nuqte ko unke nearest ligature ke saath group karta hai
        """
        grouped = []
        used_diacritics = set()
        
        for primary in primary_components:
            p_left = primary['x']
            p_right = primary['x'] + primary['w']
            p_top = primary['y']
            p_bottom = primary['y'] + primary['h']
            p_center_x = primary['x'] + primary['w'] / 2
            
            # Find associated diacritics
            associated_diacritics = []
            
            for idx, secondary in enumerate(secondary_components):
                if idx in used_diacritics:
                    continue
                
                s_center_x = secondary['x'] + secondary['w'] / 2
                s_top = secondary['y']
                s_bottom = secondary['y'] + secondary['h']
                
                # Horizontal overlap check (diacritic should be roughly above/below ligature)
                horizontal_overlap = (p_left - 15 <= s_center_x <= p_right + 15)
                
                # Vertical proximity check (not too far above or below)
                vertical_distance_top = abs(s_bottom - p_top)
                vertical_distance_bottom = abs(s_top - p_bottom)
                vertical_proximity = (vertical_distance_top < 40 or vertical_distance_bottom < 40)
                
                if horizontal_overlap and vertical_proximity:
                    associated_diacritics.append(secondary)
                    used_diacritics.add(idx)
            
            # Combine bounding boxes
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
                    'primary': primary,
                    'diacritics': associated_diacritics,
                    'diacritic_count': len(associated_diacritics)
                })
            else:
                grouped.append({
                    'x': primary['x'], 'y': primary['y'],
                    'w': primary['w'], 'h': primary['h'],
                    'area': primary['area'],
                    'primary': primary,
                    'diacritics': [],
                    'diacritic_count': 0
                })
        
        # Add orphan diacritics (jo kisi ligature se match nahi huay)
        orphan_diacritics = [secondary_components[i] for i in range(len(secondary_components)) 
                            if i not in used_diacritics]
        
        return grouped, orphan_diacritics
    
    def detect_and_classify(self, binary_image):
        """Detect components and classify them"""
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
                components.append({
                    'x': x, 'y': y, 'w': w, 'h': h, 
                    'area': area, 'cy': cy
                })
        
        if not components:
            return [], [], []
        
        # Baseline detection
        h_proj = np.sum(binary_image, axis=1)
        baseline = np.argmax(h_proj)
        
        # Classify
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
        grouped_ligatures, orphan_diacritics = self.group_diacritics_with_ligatures(
            primary_components, secondary_components
        )
        
        return grouped_ligatures, orphan_diacritics, components
    
    def process_image(self, image_path):
        """Process single image"""
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(image_path)}")
        print(f"{'='*60}")
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"  ✗ Could not load image")
            return None
        
        print(f"  Size: {image.shape[1]}x{image.shape[0]}")
        print(f"  📌 ENHANCED PREPROCESSING:")
        print(f"    ↳ Contrast Enhancement (CLAHE)...")
        print(f"    ↳ Adaptive Thresholding...")
        print(f"    ↳ Noise Removal (Morphological Ops)...")
        
        binary = self.preprocess_image(image)
        
        # Save binary
        binary_path = os.path.join(self.binary_folder, f"{base_name}_binary.png")
        cv2.imwrite(binary_path, binary)
        print(f"  ✓ Binary saved: {os.path.basename(binary_path)}")
        
        # Detect and group
        grouped_ligatures, orphan_diacritics, all_components = self.detect_and_classify(binary)
        
        print(f"  ✓ Components found: {len(all_components)}")
        print(f"  ✓ Grouped ligatures (with diacritics): {len(grouped_ligatures)}")
        total_diacritics = sum(g['diacritic_count'] for g in grouped_ligatures)
        print(f"  ✓ Diacritics grouped with ligatures: {total_diacritics}")
        print(f"  ✓ Orphan diacritics: {len(orphan_diacritics)}")
        
        # Visualization
        vis_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        for group in grouped_ligatures:
            x, y, w, h = group['x'], group['y'], group['w'], group['h']
            # Green box = ligature WITH diacritics
            color = (0, 255, 0) if group['diacritic_count'] > 0 else (255, 0, 0)
            cv2.rectangle(vis_image, (x, y), (x+w, y+h), color, 2)
            
            # Label
            label = f"{group['diacritic_count']}" if group['diacritic_count'] > 0 else ""
            if label:
                cv2.putText(vis_image, label, (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        for orphan in orphan_diacritics:
            x, y, w, h = orphan['x'], orphan['y'], orphan['w'], orphan['h']
            cv2.rectangle(vis_image, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Red = orphan
        
        vis_path = os.path.join(self.visualization_folder, f"{base_name}_grouped_visualization.png")
        cv2.imwrite(vis_path, vis_image)
        print(f"  ✓ Visualization saved: {os.path.basename(vis_path)}")
        
        # Save grouped characters
        char_folder = os.path.join(self.characters_folder, base_name)
        os.makedirs(char_folder, exist_ok=True)
        
        for idx, group in enumerate(grouped_ligatures):
            x, y, w, h = group['x'], group['y'], group['w'], group['h']
            char_img = binary[y:y+h, x:x+w]
            char_path = os.path.join(char_folder, f"char_{idx:03d}_with_{group['diacritic_count']}_dots.png")
            cv2.imwrite(char_path, char_img)
        
        print(f"  ✓ Grouped characters saved to: {base_name}/")
        
        return {
            'image_name': os.path.basename(image_path),
            'grouped_ligatures': len(grouped_ligatures),
            'total_diacritics_grouped': total_diacritics,
            'orphan_diacritics': len(orphan_diacritics),
            'total_components': len(all_components)
        }
    
    def process_all_images(self):
        """Process all images in input folder"""
        image_files = [f for f in os.listdir(self.input_folder)
                      if os.path.splitext(f.lower())[1] in self.supported_formats]
        
        if not image_files:
            print(f"\n✗ No images found in '{self.input_folder}'!")
            return []
        
        print(f"\n{'='*60}")
        print("BATCH URDU OCR SYSTEM (WITH DIACRITIC GROUPING)")
        print(f"{'='*60}")
        print(f"\n✓ Found {len(image_files)} images")
        print(f"  Input: {self.input_folder}")
        print(f"  Output: {self.output_folder}")
        
        results = []
        for image_file in image_files:
            image_path = os.path.join(self.input_folder, image_file)
            result = self.process_image(image_path)
            if result:
                results.append(result)
        
        # Summary
        print("\n" + "="*60)
        print("PROCESSING COMPLETE!")
        print("="*60)
        print(f"\n✓ Images processed: {len(results)}/{len(image_files)}")
        
        if results:
            total_grouped = sum(r['grouped_ligatures'] for r in results)
            total_diacritics = sum(r['total_diacritics_grouped'] for r in results)
            total_orphans = sum(r['orphan_diacritics'] for r in results)
            
            print(f"  Grouped ligatures (with dots): {total_grouped}")
            print(f"  Diacritics grouped: {total_diacritics}")
            print(f"  Orphan diacritics: {total_orphans}")
        
        print(f"\n📁 Color Code:")
        print(f"   🟢 Green = Ligature WITH diacritics")
        print(f"   🔵 Blue = Ligature WITHOUT diacritics")
        print(f"   🔴 Red = Orphan diacritics")
        
        return results


def main():
    import sys
    
    print("="*60)
    print("BATCH URDU CHARACTER DETECTION")
    print("WITH DIACRITIC GROUPING")
    print("="*60)
    
    input_folder = "input_images"
    output_folder = "output_grouped"
    
    if len(sys.argv) > 1:
        input_folder = sys.argv[1]
    if len(sys.argv) > 2:
        output_folder = sys.argv[2]
    
    processor = BatchUrduOCRGrouped(input_folder=input_folder, output_folder=output_folder)
    processor.process_all_images()


if __name__ == "__main__":
    main()
