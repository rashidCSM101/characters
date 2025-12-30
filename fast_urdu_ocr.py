"""
Fast Urdu OCR - Ligatures with Dots
===================================
Optimized version for faster processing
Dots are associated with their ligatures (شد with 3 dots = 1 image)
"""

import cv2
import numpy as np
import os


class FastUrduOCR:
    def __init__(self, input_folder="input_images", output_folder="output"):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.binary_folder = os.path.join(output_folder, "binary_images")
        self.visualization_folder = os.path.join(output_folder, "visualization_images")
        self.characters_folder = os.path.join(output_folder, "extracted_characters")
        
        for folder in [self.input_folder, self.output_folder, self.binary_folder,
                       self.visualization_folder, self.characters_folder]:
            os.makedirs(folder, exist_ok=True)
    
    def process_image(self, image_path):
        """Process single image - extract ligatures with their dots"""
        
        # Get filename
        filename = os.path.basename(image_path)
        base_name = os.path.splitext(filename)[0]
        
        print(f"\nProcessing: {filename}")
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"  Error loading image!")
            return
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Binarization (Otsu)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Save binary
        binary_path = os.path.join(self.binary_folder, f"{base_name}_binary.png")
        cv2.imwrite(binary_path, binary)
        print(f"  Saved binary image")
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        print(f"  Found {num_labels - 1} components")
        
        # Separate primary (ligatures) and secondary (dots)
        components = []
        for i in range(1, num_labels):
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            cx, cy = centroids[i]
            
            if area > 5:  # Filter noise
                components.append({
                    'id': i, 'x': x, 'y': y, 'w': w, 'h': h,
                    'area': area, 'cx': cx, 'cy': cy
                })
        
        if not components:
            print("  No components found!")
            return
        
        # Calculate thresholds for primary/secondary separation
        areas = [c['area'] for c in components]
        heights = [c['h'] for c in components]
        median_area = np.median(areas)
        median_height = np.median(heights)
        
        primary = []
        secondary = []
        
        for c in components:
            # Small components are dots (diacritics)
            is_dot = c['area'] < median_area * 0.25 or c['h'] < median_height * 0.35
            
            if is_dot:
                secondary.append(c)
            else:
                primary.append(c)
        
        print(f"  Primary ligatures: {len(primary)}")
        print(f"  Diacritics (dots): {len(secondary)}")
        
        # Associate dots with nearest ligature (horizontally)
        for p in primary:
            p['dots'] = []
        
        for dot in secondary:
            best_ligature = None
            best_score = -float('inf')
            
            for p in primary:
                # Check if dot is horizontally within ligature bounds
                margin = p['w'] * 0.3
                if (p['x'] - margin) <= dot['cx'] <= (p['x'] + p['w'] + margin):
                    # Calculate distance score
                    h_dist = abs(dot['cx'] - (p['x'] + p['w']/2))
                    score = 100 - h_dist  # Higher score for closer dots
                    
                    if score > best_score:
                        best_score = score
                        best_ligature = p
            
            if best_ligature:
                best_ligature['dots'].append(dot)
        
        # Create output folder for this image
        img_folder = os.path.join(self.characters_folder, base_name)
        lig_folder = os.path.join(img_folder, "ligatures_with_dots")
        os.makedirs(lig_folder, exist_ok=True)
        
        # Sort right to left (Urdu reading order)
        primary_sorted = sorted(primary, key=lambda c: -c['x'])
        
        # Extract and save each ligature with its dots
        vis_img = img.copy()
        
        for idx, lig in enumerate(primary_sorted):
            # Get bounding box including dots
            x1, y1 = lig['x'], lig['y']
            x2, y2 = lig['x'] + lig['w'], lig['y'] + lig['h']
            
            for dot in lig['dots']:
                x1 = min(x1, dot['x'])
                y1 = min(y1, dot['y'])
                x2 = max(x2, dot['x'] + dot['w'])
                y2 = max(y2, dot['y'] + dot['h'])
            
            # Add padding
            pad = 3
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(binary.shape[1], x2 + pad)
            y2 = min(binary.shape[0], y2 + pad)
            
            # Crop ligature with dots
            lig_img = binary[y1:y2, x1:x2]
            
            # Save
            num_dots = len(lig['dots'])
            filepath = os.path.join(lig_folder, f"lig_{idx+1:04d}_dots{num_dots}.png")
            cv2.imwrite(filepath, lig_img)
            
            # Draw on visualization
            color = (0, 255, 0) if num_dots == 0 else (0, 165, 255) if num_dots <= 2 else (0, 0, 255)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis_img, f"{num_dots}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Save visualization
        viz_path = os.path.join(self.visualization_folder, f"{base_name}_viz.png")
        cv2.imwrite(viz_path, vis_img)
        
        print(f"  Saved {len(primary_sorted)} ligatures to: {lig_folder}")
        print(f"  Saved visualization")
        
        return len(primary_sorted)
    
    def process_all(self):
        """Process all images in input folder"""
        print("="*50)
        print("URDU OCR - LIGATURES WITH DOTS")
        print("="*50)
        
        # Get all images
        extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        images = [f for f in os.listdir(self.input_folder) 
                  if os.path.splitext(f)[1].lower() in extensions]
        
        if not images:
            print(f"\nNo images in '{self.input_folder}'!")
            return
        
        print(f"\nFound {len(images)} images")
        
        total_ligatures = 0
        for img_file in images:
            img_path = os.path.join(self.input_folder, img_file)
            count = self.process_image(img_path)
            if count:
                total_ligatures += count
        
        print("\n" + "="*50)
        print("COMPLETE!")
        print(f"Total ligatures extracted: {total_ligatures}")
        print("="*50)
        print(f"\nOutput Structure:")
        print(f"  {self.output_folder}/")
        print(f"  ├── binary_images/")
        print(f"  ├── visualization_images/")
        print(f"  └── extracted_characters/")
        print(f"      └── [image_name]/")
        print(f"          └── ligatures_with_dots/")
        print(f"              ├── lig_0001_dots3.png  ← شد with 3 dots")
        print(f"              ├── lig_0002_dots1.png  ← ب with 1 dot")
        print(f"              └── lig_0003_dots0.png  ← ا with no dots")


if __name__ == "__main__":
    import sys
    
    input_folder = sys.argv[1] if len(sys.argv) > 1 else "input_images"
    output_folder = sys.argv[2] if len(sys.argv) > 2 else "output"
    
    ocr = FastUrduOCR(input_folder, output_folder)
    ocr.process_all()
