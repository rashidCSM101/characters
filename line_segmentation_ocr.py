"""
Line Segmentation + OCR System
================================
Pehle lines alag karenge, phir har line pe OCR chalayenge
"""

import cv2
import numpy as np
import os
import sys
import json


class LineSegmentationOCR:
    """
    Step 1: Lines extract karta hai (horizontal projection)
    Step 2: Har line pe separately OCR karta hai
    """
    
    def __init__(self, output_folder="output_line_segmented"):
        self.output_folder = output_folder
        self.lines_folder = os.path.join(output_folder, "segmented_lines")
        self.results_folder = os.path.join(output_folder, "line_results")
        
        for folder in [output_folder, self.lines_folder, self.results_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)
        
        self.min_component_area = 3
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
        """Enhanced preprocessing"""
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
    
    def segment_lines(self, binary_image, original_image):
        """
        🎯 KEY FUNCTION: Horizontal projection se lines alag karta hai
        """
        height, width = binary_image.shape
        
        # Horizontal projection (har row mein kitne pixels hain)
        h_projection = np.sum(binary_image, axis=1) / 255
        
        # Smooth karo (noise reduce)
        kernel_size = 5
        h_projection_smooth = np.convolve(h_projection, 
                                         np.ones(kernel_size)/kernel_size, 
                                         mode='same')
        
        # Threshold - text hai ya nahi
        threshold = np.mean(h_projection_smooth) * 0.3
        
        # Find text regions
        in_text = False
        line_starts = []
        line_ends = []
        
        for i in range(len(h_projection_smooth)):
            if h_projection_smooth[i] > threshold and not in_text:
                line_starts.append(i)
                in_text = True
            elif h_projection_smooth[i] <= threshold and in_text:
                line_ends.append(i)
                in_text = False
        
        # Last line handle
        if in_text:
            line_ends.append(height)
        
        # Extract lines with padding
        lines = []
        padding = 5
        
        for start, end in zip(line_starts, line_ends):
            # Add padding
            start_padded = max(0, start - padding)
            end_padded = min(height, end + padding)
            
            # Extract line from original and binary
            line_binary = binary_image[start_padded:end_padded, :]
            line_original = original_image[start_padded:end_padded, :]
            
            # Skip if too small
            if line_binary.shape[0] < 10:
                continue
            
            lines.append({
                'binary': line_binary,
                'original': line_original,
                'start': start_padded,
                'end': end_padded,
                'height': end_padded - start_padded
            })
        
        return lines, h_projection_smooth
    
    def group_diacritics_with_ligatures(self, primary_components, secondary_components):
        """Group diacritics with ligatures"""
        grouped = []
        used_diacritics = set()
        
        for primary in primary_components:
            p_left = primary['x']
            p_right = primary['x'] + primary['w']
            p_top = primary['y']
            p_bottom = primary['y'] + primary['h']
            
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
            
            grouped.append({
                'x': primary['x'], 'y': primary['y'],
                'w': primary['w'], 'h': primary['h'],
                'diacritic_count': len(associated_diacritics)
            })
        
        orphan_diacritics = [secondary_components[i] for i in range(len(secondary_components)) 
                            if i not in used_diacritics]
        
        return grouped, orphan_diacritics
    
    def detect_in_line(self, line_binary):
        """Ek line mein characters detect karo"""
        # Connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            line_binary, connectivity=8
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
            return 0, 0, 0
        
        # Baseline detection
        h_proj = np.sum(line_binary, axis=1)
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
        
        # Group
        grouped_ligatures, orphan_diacritics = self.group_diacritics_with_ligatures(
            primary_components, secondary_components
        )
        
        return len(grouped_ligatures), len(orphan_diacritics), len(components)
    
    def process_image(self, image_path, first_line_only=True):
        """
        Main processing function
        first_line_only=True: Sirf pehli line process karo
        """
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        print(f"\n{'='*70}")
        print(f"Processing: {os.path.basename(image_path)}")
        print(f"{'='*70}")
        
        # Load image
        original = cv2.imread(image_path)
        if original is None:
            print(f"  ✗ Could not load image")
            return None
        
        print(f"  Size: {original.shape[1]}x{original.shape[0]}")
        
        # Preprocess
        print(f"  📌 PREPROCESSING...")
        binary = self.preprocess_image(original)
        
        # Segment lines
        print(f"  📌 LINE SEGMENTATION...")
        lines, h_projection = self.segment_lines(binary, original)
        
        print(f"  ✓ Found {len(lines)} line(s)")
        
        if len(lines) == 0:
            print("  ✗ No lines detected!")
            return None
        
        # Save segmented lines
        for idx, line in enumerate(lines):
            line_path = os.path.join(self.lines_folder, 
                                    f"{base_name}_line_{idx+1}.png")
            cv2.imwrite(line_path, line['binary'])
            print(f"  ✓ Line {idx+1} saved: {os.path.basename(line_path)}")
        
        # Process lines
        if first_line_only:
            print(f"\n  🎯 Processing FIRST LINE only...")
            lines_to_process = [lines[0]]
            start_idx = 1
        else:
            print(f"\n  🎯 Processing ALL {len(lines)} lines...")
            lines_to_process = lines
            start_idx = 1
        
        results = []
        
        for idx, line in enumerate(lines_to_process, start=start_idx):
            print(f"\n  ➤ Line {idx}:")
            
            ligatures, orphans, total = self.detect_in_line(line['binary'])
            
            print(f"    ├─ Grouped Ligatures: {ligatures}")
            print(f"    ├─ Orphan Diacritics: {orphans}")
            print(f"    └─ Total Components: {total}")
            
            # Visualization
            vis_image = cv2.cvtColor(line['binary'], cv2.COLOR_GRAY2BGR)
            
            # Save visualization
            vis_path = os.path.join(self.results_folder, 
                                   f"{base_name}_line_{idx}_result.png")
            cv2.imwrite(vis_path, vis_image)
            
            results.append({
                'line_number': idx,
                'ligatures': ligatures,
                'orphans': orphans,
                'total': total
            })
        
        # Summary
        print(f"\n{'='*70}")
        print("SUMMARY:")
        print(f"{'='*70}")
        
        total_ligatures = sum(r['ligatures'] for r in results)
        total_orphans = sum(r['orphans'] for r in results)
        total_components = sum(r['total'] for r in results)
        
        print(f"  Lines processed: {len(results)}")
        print(f"  Total Ligatures: {total_ligatures}")
        print(f"  Total Orphan Diacritics: {total_orphans}")
        print(f"  Total Components: {total_components}")
        
        # Save results
        result_file = os.path.join(self.results_folder, f"{base_name}_results.json")
        with open(result_file, 'w') as f:
            json.dump({
                'image': os.path.basename(image_path),
                'total_lines': len(lines),
                'processed_lines': len(results),
                'results': results,
                'totals': {
                    'ligatures': total_ligatures,
                    'orphans': total_orphans,
                    'components': total_components
                }
            }, f, indent=2)
        
        print(f"\n  ✅ Results saved: {result_file}")
        
        return results


def main():
    print("="*70)
    print("LINE SEGMENTATION + OCR SYSTEM")
    print("="*70)
    
    if len(sys.argv) < 2:
        print("\n📝 Usage:")
        print("   python line_segmentation_ocr.py <image_path> [first_line_only]")
        print("\n📌 Examples:")
        print('   python line_segmentation_ocr.py "input_images/4 (124).png" True')
        print('   python line_segmentation_ocr.py "input_images/4 (124).png" False')
        print("\n❓ Parameters:")
        print("   first_line_only = True: Sirf pehli line process karo (DEFAULT)")
        print("   first_line_only = False: Sari lines process karo")
        return
    
    image_path = sys.argv[1]
    first_line_only = True
    
    if len(sys.argv) > 2:
        first_line_only = sys.argv[2].lower() in ['true', '1', 'yes']
    
    if not os.path.exists(image_path):
        print(f"❌ Error: File not found: {image_path}")
        return
    
    print(f"\n📋 Settings:")
    print(f"   Image: {image_path}")
    print(f"   Process first line only: {first_line_only}")
    
    processor = LineSegmentationOCR()
    processor.process_image(image_path, first_line_only=first_line_only)


if __name__ == "__main__":
    main()
