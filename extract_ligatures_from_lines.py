"""
Extract Ligatures from Line-Segmented Images
=============================================
Line segmentation ki output files input leta hai
NO preprocessing - direct binary use karta hai
Ligatures extract kar ke separate folder mein save karta hai
"""

import cv2
import numpy as np
import os
import sys
import json
from datetime import datetime


class LigatureExtractor:
    """
    Line segmented images se ligatures extract karta hai
    NO preprocessing - images already processed hain
    """
    
    def __init__(self, input_folder="output_line_segmented/segmented_lines", 
                 output_folder="output_ligatures"):
        self.input_folder = input_folder
        self.output_folder = output_folder
        
        # Output folders
        self.ligatures_folder = os.path.join(output_folder, "ligatures")
        self.stats_folder = os.path.join(output_folder, "stats")
        
        # Create folders
        for folder in [output_folder, self.ligatures_folder, self.stats_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)
        
        self.min_component_area = 3
    
    def group_diacritics_with_ligatures(self, primary_components, secondary_components):
        """Group diacritics with their nearest primary ligature"""
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
            
            # Calculate bounding box for grouped ligature
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
    
    def extract_ligatures_from_line(self, line_image_path):
        """
        Line image se ligatures extract karta hai
        NO preprocessing - direct binary use
        """
        # Load image
        image = cv2.imread(line_image_path)
        if image is None:
            print(f"  ❌ Could not load: {line_image_path}")
            return None
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Check if already binary (most pixels are 0 or 255)
        unique_vals = np.unique(gray)
        is_binary = len(unique_vals) <= 10  # Already binary or near-binary
        
        if is_binary:
            # Already binary - just invert if needed
            # Check if background is white (most common value is 255)
            if np.median(gray) > 127:
                binary = cv2.bitwise_not(gray)
            else:
                binary = gray.copy()
        else:
            # Not binary - apply Otsu thresholding
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Connected component labeling
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
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
                    'x': x, 'y': y, 'w': w, 'h': h, 'area': area, 'cy': cy
                })
        
        if not components:
            return {
                'grouped_ligatures': [],
                'orphan_diacritics': [],
                'total_ligatures': 0,
                'total_orphans': 0,
                'total_components': 0
            }
        
        # Baseline detection
        h_proj = np.sum(binary, axis=1)
        baseline = np.argmax(h_proj)
        
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
        grouped_ligatures, orphan_diacritics = self.group_diacritics_with_ligatures(
            primary_components, secondary_components
        )
        
        return {
            'binary': binary,
            'grouped_ligatures': grouped_ligatures,
            'orphan_diacritics': orphan_diacritics,
            'total_ligatures': len(grouped_ligatures),
            'total_orphans': len(orphan_diacritics),
            'total_components': len(components)
        }
    
    def save_ligature_images(self, binary_image, ligatures, base_name, line_num):
        """Save individual ligature images"""
        line_folder = os.path.join(self.ligatures_folder, base_name, f"line_{line_num}")
        os.makedirs(line_folder, exist_ok=True)
        
        saved_count = 0
        for idx, lig in enumerate(ligatures, 1):
            x, y, w, h = lig['x'], lig['y'], lig['w'], lig['h']
            
            # Extract ligature
            ligature_img = binary_image[y:y+h, x:x+w]
            
            # Save
            filename = f"ligature_{idx:03d}.png"
            filepath = os.path.join(line_folder, filename)
            cv2.imwrite(filepath, ligature_img)
            saved_count += 1
        
        return saved_count
    
    def process_all_lines(self, image_base_name):
        """
        Process all line images for a given base image name
        Example: test.png -> test_line_1.png, test_line_2.png, etc.
        """
        print(f"\n{'='*70}")
        print(f"Processing lines for: {image_base_name}")
        print(f"{'='*70}")
        
        # Find all line images
        line_files = []
        for filename in os.listdir(self.input_folder):
            if filename.startswith(image_base_name) and filename.endswith('.png'):
                line_files.append(filename)
        
        if not line_files:
            print(f"❌ No line images found for: {image_base_name}")
            return
        
        # Sort by line number
        line_files.sort()
        
        print(f"✓ Found {len(line_files)} line image(s)")
        
        # Process each line
        all_results = []
        total_ligatures = 0
        total_orphans = 0
        total_components = 0
        
        for line_file in line_files:
            line_path = os.path.join(self.input_folder, line_file)
            line_num = line_file.replace(image_base_name + "_line_", "").replace(".png", "")
            
            print(f"\n  ➤ Line {line_num}:")
            
            result = self.extract_ligatures_from_line(line_path)
            
            if result is None:
                continue
            
            print(f"    ├─ Grouped Ligatures: {result['total_ligatures']}")
            print(f"    ├─ Orphan Diacritics: {result['total_orphans']}")
            print(f"    └─ Total Components: {result['total_components']}")
            
            # Save ligature images
            if result['grouped_ligatures']:
                saved = self.save_ligature_images(
                    result['binary'], 
                    result['grouped_ligatures'], 
                    image_base_name, 
                    line_num
                )
                print(f"    ✓ Saved {saved} ligature images")
            
            all_results.append({
                'line': line_num,
                'file': line_file,
                'ligatures': result['total_ligatures'],
                'orphans': result['total_orphans'],
                'components': result['total_components']
            })
            
            total_ligatures += result['total_ligatures']
            total_orphans += result['total_orphans']
            total_components += result['total_components']
        
        # Summary
        print(f"\n{'='*70}")
        print("SUMMARY:")
        print(f"{'='*70}")
        print(f"  Lines processed: {len(line_files)}")
        print(f"  Total Ligatures: {total_ligatures}")
        print(f"  Total Orphan Diacritics: {total_orphans}")
        print(f"  Total Components: {total_components}")
        
        # Save stats
        stats = {
            'image_base_name': image_base_name,
            'total_lines': len(line_files),
            'total_ligatures': total_ligatures,
            'total_orphans': total_orphans,
            'total_components': total_components,
            'lines': all_results,
            'processed_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        stats_file = os.path.join(self.stats_folder, f"{image_base_name}_stats.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"\n  ✅ Stats saved: {stats_file}")
        print(f"  ✅ Ligatures saved: {os.path.join(self.ligatures_folder, image_base_name)}")


def main():
    print("="*70)
    print("LIGATURE EXTRACTOR FROM LINE-SEGMENTED IMAGES")
    print("="*70)
    print("\n📝 NO PREPROCESSING - Direct extraction from line images")
    
    if len(sys.argv) < 2:
        print("\n📝 Usage:")
        print("   python extract_ligatures_from_lines.py <image_base_name>")
        print("\n📌 Examples:")
        print('   python extract_ligatures_from_lines.py test')
        print('   python extract_ligatures_from_lines.py "4 (124)"')
        print("\n❓ Note:")
        print("   - Reads from: output_line_segmented/segmented_lines/")
        print("   - Looks for: <base_name>_line_1.png, <base_name>_line_2.png, etc.")
        print("   - Saves to: output_ligatures/")
        return
    
    base_name = sys.argv[1]
    
    # Initialize extractor
    extractor = LigatureExtractor()
    
    # Process all lines
    extractor.process_all_lines(base_name)


if __name__ == "__main__":
    main()
