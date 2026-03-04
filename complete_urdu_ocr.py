"""
COMPLETE URDU OCR SYSTEM
========================
Ek file mein sab kuch:
1. Enhanced Preprocessing
2. Line Segmentation  
3. Character Detection (with diacritics grouped)
4. Accuracy Calculation
5. Results + Visualization

Usage: python complete_urdu_ocr.py <image_path>
"""

import cv2
import numpy as np
import os
import json
from datetime import datetime


class CompleteUrduOCR:
    """All-in-one Urdu OCR System"""
    
    def __init__(self, image_path):
        self.image_path = image_path
        self.base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Output folders
        self.output_folder = f"results_{self.base_name}"
        self.lines_folder = os.path.join(self.output_folder, "1_segmented_lines")
        self.binary_folder = os.path.join(self.output_folder, "2_binary_images")
        self.vis_folder = os.path.join(self.output_folder, "3_visualizations")
        self.chars_folder = os.path.join(self.output_folder, "4_extracted_chars")
        
        # Create folders
        for folder in [self.output_folder, self.lines_folder, self.binary_folder, 
                      self.vis_folder, self.chars_folder]:
            os.makedirs(folder, exist_ok=True)
        
        self.min_area = 3
    
    # ====================================================================
    # STEP 1: PREPROCESSING
    # ====================================================================
    
    def preprocess(self, image):
        """Enhanced preprocessing"""
        # Grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 1. Bilateral Denoising
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # 2. CLAHE Contrast Enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # 3. Adaptive Thresholding
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 13, 3
        )
        
        # 4. Morphological Noise Removal
        kernel_tiny = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_tiny)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_small)
        
        return binary
    
    # ====================================================================
    # STEP 2: LINE SEGMENTATION
    # ====================================================================
    
    def segment_lines(self, binary_image):
        """Extract text lines using horizontal projection"""
        height = binary_image.shape[0]
        
        # Horizontal projection
        h_proj = np.sum(binary_image, axis=1) / 255
        
        # Smooth
        h_proj_smooth = np.convolve(h_proj, np.ones(5)/5, mode='same')
        
        # Threshold
        threshold = np.mean(h_proj_smooth) * 0.3
        
        # Find line boundaries
        in_line = False
        lines = []
        start = 0
        
        for i in range(len(h_proj_smooth)):
            if h_proj_smooth[i] > threshold and not in_line:
                start = i
                in_line = True
            elif h_proj_smooth[i] <= threshold and in_line:
                end = i
                # Add padding
                start_pad = max(0, start - 5)
                end_pad = min(height, end + 5)
                
                if end_pad - start_pad > 10:  # Min height
                    lines.append({
                        'start': start_pad,
                        'end': end_pad,
                        'height': end_pad - start_pad
                    })
                in_line = False
        
        # Handle last line
        if in_line:
            end_pad = min(height, len(h_proj_smooth) + 5)
            if end_pad - start > 10:
                lines.append({
                    'start': max(0, start - 5),
                    'end': end_pad,
                    'height': end_pad - start
                })
        
        return lines
    
    # ====================================================================
    # STEP 3: CHARACTER DETECTION
    # ====================================================================
    
    def detect_in_line(self, line_binary):
        """Detect and group characters in a single line"""
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
            
            if area >= self.min_area:
                components.append({
                    'x': x, 'y': y, 'w': w, 'h': h, 
                    'area': area, 'cy': cy
                })
        
        if not components:
            return [], [], []
        
        # Find baseline
        h_proj = np.sum(line_binary, axis=1)
        baseline = np.argmax(h_proj)
        
        # Classify components
        heights = [c['h'] for c in components]
        median_height = np.median(heights)
        threshold = median_height * 0.5
        
        primary = []
        secondary = []
        
        for comp in components:
            if abs(comp['cy'] - baseline) > threshold:
                secondary.append(comp)
            else:
                primary.append(comp)
        
        # Group diacritics with ligatures
        grouped = []
        used = set()
        
        for p in primary:
            p_left = p['x']
            p_right = p['x'] + p['w']
            p_top = p['y']
            p_bottom = p['y'] + p['h']
            
            associated = []
            for idx, s in enumerate(secondary):
                if idx in used:
                    continue
                
                s_cx = s['x'] + s['w'] / 2
                s_top = s['y']
                s_bottom = s['y'] + s['h']
                
                # Check overlap
                h_overlap = (p_left - 15 <= s_cx <= p_right + 15)
                v_dist_top = abs(s_bottom - p_top)
                v_dist_bottom = abs(s_top - p_bottom)
                v_near = (v_dist_top < 40 or v_dist_bottom < 40)
                
                if h_overlap and v_near:
                    associated.append(s)
                    used.add(idx)
            
            # Create grouped component
            if associated:
                all_comps = [p] + associated
                x_min = min(c['x'] for c in all_comps)
                y_min = min(c['y'] for c in all_comps)
                x_max = max(c['x'] + c['w'] for c in all_comps)
                y_max = max(c['y'] + c['h'] for c in all_comps)
                
                grouped.append({
                    'x': x_min, 'y': y_min,
                    'w': x_max - x_min, 'h': y_max - y_min,
                    'diacritics': len(associated),
                    'has_diacritics': True
                })
            else:
                grouped.append({
                    'x': p['x'], 'y': p['y'],
                    'w': p['w'], 'h': p['h'],
                    'diacritics': 0,
                    'has_diacritics': False
                })
        
        # Orphan diacritics
        orphans = [secondary[i] for i in range(len(secondary)) if i not in used]
        
        return grouped, orphans, components
    
    # ====================================================================
    # STEP 4: VISUALIZATION
    # ====================================================================
    
    def visualize_line(self, line_binary, grouped, orphans):
        """Create visualization with colored boxes"""
        vis = cv2.cvtColor(line_binary, cv2.COLOR_GRAY2BGR)
        
        # Grouped ligatures
        for g in grouped:
            x, y, w, h = g['x'], g['y'], g['w'], g['h']
            color = (0, 255, 0) if g['has_diacritics'] else (255, 0, 0)
            cv2.rectangle(vis, (x, y), (x+w, y+h), color, 2)
            
            # Label with diacritic count
            if g['diacritics'] > 0:
                cv2.putText(vis, str(g['diacritics']), (x, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Orphan diacritics
        for o in orphans:
            x, y, w, h = o['x'], o['y'], o['w'], o['h']
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        return vis
    
    # ====================================================================
    # MAIN PROCESSING
    # ====================================================================
    
    def process(self):
        """Main processing pipeline"""
        print("="*70)
        print(f"URDU OCR - Processing: {self.base_name}")
        print("="*70)
        
        # Load image
        original = cv2.imread(self.image_path)
        if original is None:
            print("❌ Error: Could not load image!")
            return None
        
        print(f"\n✓ Image loaded: {original.shape[1]}x{original.shape[0]}")
        
        # Preprocess
        print("📌 Step 1: Preprocessing...")
        binary = self.preprocess(original)
        
        # Save full binary
        cv2.imwrite(os.path.join(self.binary_folder, f"{self.base_name}_full.png"), binary)
        print("  ✓ Binary image saved")
        
        # Segment lines
        print("\n📌 Step 2: Line Segmentation...")
        lines = self.segment_lines(binary)
        print(f"  ✓ Found {len(lines)} line(s)")
        
        if not lines:
            print("❌ No lines detected!")
            return None
        
        # Process each line
        print("\n📌 Step 3: Character Detection...")
        all_results = []
        
        for idx, line_info in enumerate(lines, 1):
            line_binary = binary[line_info['start']:line_info['end'], :]
            
            # Save line
            line_path = os.path.join(self.lines_folder, f"line_{idx}.png")
            cv2.imwrite(line_path, line_binary)
            
            # Detect
            grouped, orphans, all_comps = self.detect_in_line(line_binary)
            
            print(f"\n  Line {idx}:")
            print(f"    ├─ Ligatures (with diacritics): {len(grouped)}")
            print(f"    ├─ Orphan diacritics: {len(orphans)}")
            print(f"    └─ Total components: {len(all_comps)}")
            
            # Visualize
            vis = self.visualize_line(line_binary, grouped, orphans)
            vis_path = os.path.join(self.vis_folder, f"line_{idx}_result.png")
            cv2.imwrite(vis_path, vis)
            
            # Save individual characters
            line_chars_folder = os.path.join(self.chars_folder, f"line_{idx}")
            os.makedirs(line_chars_folder, exist_ok=True)
            
            for char_idx, g in enumerate(grouped):
                x, y, w, h = g['x'], g['y'], g['w'], g['h']
                char_img = line_binary[y:y+h, x:x+w]
                char_name = f"char_{char_idx:03d}_dots_{g['diacritics']}.png"
                cv2.imwrite(os.path.join(line_chars_folder, char_name), char_img)
            
            all_results.append({
                'line': idx,
                'ligatures': len(grouped),
                'ligatures_with_diacritics': sum(1 for g in grouped if g['has_diacritics']),
                'total_diacritics_grouped': sum(g['diacritics'] for g in grouped),
                'orphan_diacritics': len(orphans),
                'total_components': len(all_comps)
            })
        
        # Summary
        print("\n" + "="*70)
        print("RESULTS SUMMARY:")
        print("="*70)
        
        total_ligatures = sum(r['ligatures'] for r in all_results)
        total_with_diacritics = sum(r['ligatures_with_diacritics'] for r in all_results)
        total_orphans = sum(r['orphan_diacritics'] for r in all_results)
        total_comps = sum(r['total_components'] for r in all_results)
        
        print(f"\nLines detected: {len(lines)}")
        print(f"Total ligatures: {total_ligatures}")
        print(f"  ├─ With diacritics: {total_with_diacritics}")
        print(f"  └─ Without diacritics: {total_ligatures - total_with_diacritics}")
        print(f"Orphan diacritics: {total_orphans}")
        print(f"Total components: {total_comps}")
        
        print(f"\n📁 Output saved to: {self.output_folder}/")
        print(f"  ├─ 1_segmented_lines/  (separated line images)")
        print(f"  ├─ 2_binary_images/     (preprocessed binary)")
        print(f"  ├─ 3_visualizations/    (detection results)")
        print(f"  └─ 4_extracted_chars/   (individual characters)")
        
        # Save JSON results
        result_data = {
            'image': os.path.basename(self.image_path),
            'timestamp': datetime.now().isoformat(),
            'total_lines': len(lines),
            'summary': {
                'total_ligatures': total_ligatures,
                'ligatures_with_diacritics': total_with_diacritics,
                'orphan_diacritics': total_orphans,
                'total_components': total_comps
            },
            'per_line_results': all_results
        }
        
        json_path = os.path.join(self.output_folder, 'results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Results JSON: {json_path}")
        
        print("\n" + "="*70)
        print("✅ PROCESSING COMPLETE!")
        print("="*70)
        
        return result_data


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("="*70)
        print("COMPLETE URDU OCR SYSTEM")
        print("="*70)
        print("\n📝 Usage:")
        print("   python complete_urdu_ocr.py <image_path>")
        print("\n📌 Example:")
        print('   python complete_urdu_ocr.py "input_images/4 (124).png"')
        print("\n🎯 Features:")
        print("  ✓ Enhanced preprocessing (CLAHE, bilateral filter)")
        print("  ✓ Automatic line segmentation")
        print("  ✓ Character detection with diacritics grouped")
        print("  ✓ Visual results with color coding")
        print("  ✓ Individual character extraction")
        print("  ✓ Detailed JSON results")
        print("\n🎨 Color Code:")
        print("  🟢 Green = Ligature WITH diacritics")
        print("  🔵 Blue = Ligature WITHOUT diacritics")
        print("  🔴 Red = Orphan diacritics")
        return
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"❌ Error: File not found: {image_path}")
        return
    
    # Process
    ocr = CompleteUrduOCR(image_path)
    ocr.process()


if __name__ == "__main__":
    main()
