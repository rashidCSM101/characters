"""
Batch Urdu OCR System - Process Multiple Images
================================================
Based on paper: "Optical Character Recognition System for Urdu Words in Nastaliq Font"
By Safia Shabbir and Imran Siddiqi, Bahria University

Features:
- Process all images from input folder
- Detect characters in each image
- Associate Diacritics (dots) with their Primary ligatures
- Save ligatures WITH their dots as single image
- Save binary images in separate folder
- Save visualization images in separate folder
"""

import cv2
import numpy as np
import os
from datetime import datetime


class BatchUrduOCR:
    """
    Batch processing system for Urdu character detection
    Diacritics are associated with their ligatures (like شد with 3 dots)
    """
    
    def __init__(self, input_folder="input_images", output_folder="output"):
        """
        Initialize the Batch Urdu OCR System
        
        Args:
            input_folder: Folder containing input images
            output_folder: Main output folder
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        
        # Create folder structure
        self.binary_folder = os.path.join(output_folder, "binary_images")
        self.visualization_folder = os.path.join(output_folder, "visualization_images")
        self.characters_folder = os.path.join(output_folder, "extracted_characters")
        
        # Create all folders
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
                print(f"Created folder: {folder}")
        
        # Parameters
        self.min_component_area = 10
        self.supported_formats = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    
    def preprocess_binarization(self, image):
        """
        Step 1: Binarization using Otsu's thresholding
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary
    
    def extract_components(self, binary_image):
        """
        Step 2: Extract ligatures using Connected Component Labeling
        """
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
                    'id': i,
                    'x': x, 'y': y,
                    'width': w, 'height': h,
                    'area': area,
                    'centroid_x': cx, 'centroid_y': cy,
                    'image': component_image,
                    'bbox': (x, y, w, h)
                })
        
        return components
    
    def detect_baseline(self, binary_image):
        """
        Step 3: Detect baseline using horizontal projection analysis
        Find the main text line area
        """
        horizontal_projection = np.sum(binary_image, axis=1)
        
        # Smooth the projection to find main text region
        kernel_size = 15
        if len(horizontal_projection) > kernel_size:
            kernel = np.ones(kernel_size) / kernel_size
            smoothed = np.convolve(horizontal_projection, kernel, mode='same')
        else:
            smoothed = horizontal_projection
        
        # Find the row with maximum pixels in smoothed projection
        baseline_y = np.argmax(smoothed)
        return baseline_y
    
    def separate_primary_secondary(self, components, baseline_y):
        """
        Step 4: Separate Primary Ligatures and Diacritics
        Using improved heuristics for better separation
        """
        primary_components = []
        secondary_components = []
        
        if not components:
            return [], []
        
        # Calculate statistics
        heights = [c['height'] for c in components]
        areas = [c['area'] for c in components]
        
        avg_height = np.mean(heights)
        avg_area = np.mean(areas)
        median_height = np.median(heights)
        median_area = np.median(areas)
        
        for comp in components:
            y_top = comp['y']
            y_bottom = comp['y'] + comp['height']
            
            # Check if component touches or crosses baseline region
            baseline_margin = 20  # Allow some margin
            touches_baseline = (y_top - baseline_margin) <= baseline_y <= (y_bottom + baseline_margin)
            
            # Use multiple criteria for primary/secondary classification
            is_large = comp['area'] > median_area * 0.5 and comp['height'] > median_height * 0.4
            is_small = comp['area'] < median_area * 0.15 or comp['height'] < median_height * 0.3
            
            # Primary ligatures: large components or those touching baseline
            if (touches_baseline and is_large) or (not is_small and comp['height'] > avg_height * 0.5):
                comp['type'] = 'primary'
                comp['position'] = 'baseline'
                primary_components.append(comp)
            else:
                comp['type'] = 'secondary'
                comp['position'] = 'above' if comp['centroid_y'] < baseline_y else 'below'
                secondary_components.append(comp)
        
        return primary_components, secondary_components
    
    def associate_diacritics_with_ligatures(self, primary_components, secondary_components):
        """
        Step 5: Associate diacritics (dots) with their primary ligatures
        Based on paper: "diacritics are associated with ligatures depending 
        upon their position information with respect to the ligature"
        
        Example: شد has 3 dots - they will be associated with this ligature
        """
        # Initialize associated diacritics list for each primary component
        for primary in primary_components:
            primary['associated_diacritics'] = []
        
        if not primary_components:
            return primary_components
        
        for diacritic in secondary_components:
            d_left = diacritic['x']
            d_right = diacritic['x'] + diacritic['width']
            d_center_x = diacritic['centroid_x']
            d_center_y = diacritic['centroid_y']
            
            best_match = None
            best_score = -1
            
            for primary in primary_components:
                p_left = primary['x']
                p_right = primary['x'] + primary['width']
                p_center_x = primary['centroid_x']
                p_center_y = primary['centroid_y']
                
                # Calculate horizontal overlap percentage
                overlap_left = max(d_left, p_left)
                overlap_right = min(d_right, p_right)
                overlap = max(0, overlap_right - overlap_left)
                overlap_percent = overlap / diacritic['width'] if diacritic['width'] > 0 else 0
                
                # Check if diacritic center is within ligature x-bounds (with margin)
                margin = primary['width'] * 0.2  # 20% margin
                center_within = (p_left - margin) <= d_center_x <= (p_right + margin)
                
                # Calculate horizontal distance score (closer is better)
                h_distance = abs(d_center_x - p_center_x)
                max_distance = max(primary['width'], 100)
                distance_score = max(0, 1 - (h_distance / max_distance))
                
                # Calculate total score
                score = 0
                if overlap_percent > 0:
                    score += overlap_percent * 2  # Overlap is very important
                if center_within:
                    score += 1.5  # Bonus for center being within bounds
                score += distance_score  # Add distance score
                
                # Prefer ligatures that are vertically close
                v_distance = abs(d_center_y - p_center_y)
                if v_distance < primary['height'] * 2:
                    score += 0.5
                
                if score > best_score:
                    best_score = score
                    best_match = primary
            
            # Associate diacritic with best matching ligature
            if best_match is not None and best_score > 0.3:
                best_match['associated_diacritics'].append(diacritic)
        
        return primary_components
    
    def extract_ligature_with_diacritics(self, primary, binary_image):
        """
        Extract ligature image including its associated diacritics
        Example: شد with its 3 dots as single image
        """
        # Get combined bounding box
        x_min = primary['x']
        y_min = primary['y']
        x_max = primary['x'] + primary['width']
        y_max = primary['y'] + primary['height']
        
        # Expand to include all associated diacritics
        for diacritic in primary.get('associated_diacritics', []):
            x_min = min(x_min, diacritic['x'])
            y_min = min(y_min, diacritic['y'])
            x_max = max(x_max, diacritic['x'] + diacritic['width'])
            y_max = max(y_max, diacritic['y'] + diacritic['height'])
        
        # Add padding
        padding = 2
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(binary_image.shape[1], x_max + padding)
        y_max = min(binary_image.shape[0], y_max + padding)
        
        # Crop the combined region
        combined_image = binary_image[y_min:y_max, x_min:x_max]
        
        return combined_image, (x_min, y_min, x_max, y_max)
    
    def create_visualization(self, original_image, components, baseline_y):
        """
        Create visualization with bounding boxes
        """
        if len(original_image.shape) == 2:
            vis_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        else:
            vis_image = original_image.copy()
        
        # Draw baseline
        cv2.line(vis_image, (0, baseline_y), (vis_image.shape[1], baseline_y), (255, 0, 0), 2)
        cv2.putText(vis_image, "Baseline", (10, baseline_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Draw bounding boxes
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
        """
        Process a single image and save all outputs
        """
        print(f"\n{'='*50}")
        print(f"Processing: {image_name}")
        print('='*50)
        
        # Load image
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"  ERROR: Could not load image!")
            return None
        
        print(f"  Image size: {original_image.shape[1]}x{original_image.shape[0]}")
        
        # Get base name without extension
        base_name = os.path.splitext(image_name)[0]
        
        # Step 1: Binarization
        binary_image = self.preprocess_binarization(original_image)
        
        # Save binary image
        binary_path = os.path.join(self.binary_folder, f"{base_name}_binary.png")
        cv2.imwrite(binary_path, binary_image)
        print(f"  Saved binary: {binary_path}")
        
        # Step 2: Extract components
        components = self.extract_components(binary_image)
        print(f"  Found {len(components)} components")
        
        if len(components) == 0:
            print("  No components found!")
            return None
        
        # Step 3: Detect baseline
        baseline_y = self.detect_baseline(binary_image)
        print(f"  Baseline at row: {baseline_y}")
        
        # Step 4: Separate primary and secondary
        primary, secondary = self.separate_primary_secondary(components, baseline_y)
        print(f"  Primary ligatures: {len(primary)}")
        print(f"  Diacritics: {len(secondary)}")
        
        # Step 5: Associate diacritics with their ligatures
        primary_with_diacritics = self.associate_diacritics_with_ligatures(primary, secondary)
        print(f"  Associated diacritics with ligatures")
        
        all_components = primary + secondary
        
        # Create visualization
        visualization = self.create_visualization(original_image, all_components, baseline_y)
        
        # Save visualization
        viz_path = os.path.join(self.visualization_folder, f"{base_name}_visualization.png")
        cv2.imwrite(viz_path, visualization)
        print(f"  Saved visualization: {viz_path}")
        
        # Create folder for this image's characters
        image_chars_folder = os.path.join(self.characters_folder, base_name)
        ligatures_folder = os.path.join(image_chars_folder, "ligatures_with_dots")
        
        for folder in [image_chars_folder, ligatures_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)
        
        # Save ligatures WITH their associated diacritics (dots)
        # Example: شد with its 3 dots as single image
        sorted_primary = sorted(primary_with_diacritics, key=lambda c: -c['x'])  # Right to left
        for idx, comp in enumerate(sorted_primary):
            num_dots = len(comp.get('associated_diacritics', []))
            
            # Extract ligature with its diacritics
            combined_image, bbox = self.extract_ligature_with_diacritics(comp, binary_image)
            
            # Filename shows number of dots
            filename = f"ligature_{idx+1:04d}_dots_{num_dots}.png"
            filepath = os.path.join(ligatures_folder, filename)
            cv2.imwrite(filepath, combined_image)
        
        print(f"  Saved {len(sorted_primary)} ligatures (with dots) to: {ligatures_folder}")
        
        return {
            'image_name': image_name,
            'primary_count': len(primary),
            'secondary_count': len(secondary),
            'total_components': len(all_components)
        }
    
    def process_all_images(self):
        """
        Process all images in the input folder
        """
        print("\n" + "#"*60)
        print("BATCH URDU OCR SYSTEM")
        print("Processing all images from input folder")
        print("#"*60)
        
        # Get all image files
        image_files = []
        for file in os.listdir(self.input_folder):
            ext = os.path.splitext(file)[1].lower()
            if ext in self.supported_formats:
                image_files.append(file)
        
        if len(image_files) == 0:
            print(f"\nNo images found in '{self.input_folder}'!")
            print(f"Please add images (.png, .jpg, .jpeg, .bmp, .tiff) to the folder.")
            return
        
        print(f"\nFound {len(image_files)} images to process")
        print(f"Input folder: {self.input_folder}")
        print(f"Output folder: {self.output_folder}")
        
        # Process each image
        results = []
        for image_file in image_files:
            image_path = os.path.join(self.input_folder, image_file)
            result = self.process_single_image(image_path, image_file)
            if result:
                results.append(result)
        
        # Print summary
        print("\n" + "#"*60)
        print("PROCESSING COMPLETE!")
        print("#"*60)
        print(f"\nTotal images processed: {len(results)}/{len(image_files)}")
        
        total_primary = sum(r['primary_count'] for r in results)
        total_secondary = sum(r['secondary_count'] for r in results)
        total_components = sum(r['total_components'] for r in results)
        
        print(f"Total primary ligatures: {total_primary}")
        print(f"Total diacritics: {total_secondary}")
        print(f"Total components: {total_components}")
        
        print(f"\n📁 Output Structure:")
        print(f"   {self.output_folder}/")
        print(f"   ├── binary_images/          ← Binary images")
        print(f"   ├── visualization_images/   ← Detection visualizations")
        print(f"   └── extracted_characters/   ← Ligatures for each image")
        print(f"       └── image_name/")
        print(f"           └── ligatures_with_dots/")
        print(f"               ├── ligature_0001_dots_3.png  ← شد with 3 dots")
        print(f"               ├── ligature_0002_dots_1.png  ← character with 1 dot")
        print(f"               └── ligature_0003_dots_0.png  ← character with no dots")
        
        return results


def main():
    """
    Main function
    """
    import sys
    
    print("="*60)
    print("BATCH URDU CHARACTER DETECTION SYSTEM")
    print("Based on paper by Shabbir & Siddiqi, 2016")
    print("="*60)
    
    # Default folders
    input_folder = "input_images"
    output_folder = "output"
    
    # Check command line arguments
    if len(sys.argv) > 1:
        input_folder = sys.argv[1]
    if len(sys.argv) > 2:
        output_folder = sys.argv[2]
    
    print(f"\nInput folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print("\nUsage: python batch_urdu_ocr.py [input_folder] [output_folder]")
    
    # Create processor
    processor = BatchUrduOCR(input_folder=input_folder, output_folder=output_folder)
    
    # Process all images
    processor.process_all_images()


if __name__ == "__main__":
    main()
