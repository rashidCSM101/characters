"""
Optical Character Recognition System for Urdu Words in Nastaliq Font
=====================================================================
Based on paper by Safia Shabbir and Imran Siddiqi, Bahria University

Steps Implemented:
1. Preprocessing - Binarization using Otsu's global thresholding
2. Ligature Extraction - Connected Component Labeling
3. Baseline Detection - Row with maximum text pixels
4. Primary/Secondary Component Separation (Main body vs Diacritics)
5. Feature Extraction - Horizontal/Vertical Projection, Upper/Lower Profile
6. Save each detected character/ligature as separate file

Reference: IJACSA, Vol. 7, No. 5, 2016
"""

import cv2
import numpy as np
import os
from datetime import datetime


class UrduOCRDetector:
    """
    Urdu Character/Ligature Detector based on the paper methodology
    """
    
    def __init__(self, output_folder="extracted_characters"):
        """
        Initialize the Urdu OCR Detector
        
        Args:
            output_folder: Folder where extracted characters will be saved
        """
        self.output_folder = output_folder
        self.primary_folder = os.path.join(output_folder, "primary_ligatures")
        self.secondary_folder = os.path.join(output_folder, "diacritics")
        self.all_components_folder = os.path.join(output_folder, "all_components")
        
        # Create output folders
        for folder in [self.output_folder, self.primary_folder, 
                       self.secondary_folder, self.all_components_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)
                print(f"Created folder: {folder}")
        
        # Parameters from paper
        self.min_component_area = 10  # Minimum area for valid component
        
    # =========================================================================
    # STEP 1: PREPROCESSING - Binarization using Otsu's Global Thresholding
    # =========================================================================
    def preprocess_binarization(self, image):
        """
        Step 1: Preprocessing - Binarization
        As per paper: "we have employed the well-known Otsu's global 
        thresholding to binarize the text image"
        
        Args:
            image: Input BGR or grayscale image
            
        Returns:
            Binary image (text pixels = 255, background = 0)
        """
        print("\n" + "="*60)
        print("STEP 1: PREPROCESSING - BINARIZATION")
        print("="*60)
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            print("  - Converted to grayscale")
        else:
            gray = image.copy()
        
        # Apply Otsu's global thresholding (as per paper)
        threshold_value, binary = cv2.threshold(
            gray, 0, 255, 
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        
        print(f"  - Applied Otsu's thresholding (threshold = {threshold_value:.1f})")
        print(f"  - Binary image size: {binary.shape[1]}x{binary.shape[0]}")
        
        return binary
    
    # =========================================================================
    # STEP 2: EXTRACTION OF LIGATURES - Connected Component Labeling
    # =========================================================================
    def extract_ligatures_connected_components(self, binary_image):
        """
        Step 2: Extraction of Ligatures using Connected Component Labeling
        As per paper: "Ligatures and diacritics are extracted from binarized 
        words using connected component labeling"
        
        Args:
            binary_image: Binarized image
            
        Returns:
            List of components with their properties
        """
        print("\n" + "="*60)
        print("STEP 2: EXTRACTION OF LIGATURES - CONNECTED COMPONENT LABELING")
        print("="*60)
        
        # Connected component labeling
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_image, connectivity=8
        )
        
        print(f"  - Found {num_labels - 1} connected components (excluding background)")
        
        components = []
        
        # Extract each component (skip label 0 which is background)
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            cx, cy = centroids[i]
            
            # Filter small noise components
            if area >= self.min_component_area:
                # Extract component mask
                component_mask = (labels == i).astype(np.uint8) * 255
                
                # Crop the component
                component_image = component_mask[y:y+h, x:x+w]
                
                components.append({
                    'id': i,
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'area': area,
                    'centroid_x': cx,
                    'centroid_y': cy,
                    'image': component_image,
                    'bbox': (x, y, w, h)
                })
        
        print(f"  - Filtered to {len(components)} valid components")
        
        return components
    
    # =========================================================================
    # STEP 3: BASELINE DETECTION
    # =========================================================================
    def detect_baseline(self, binary_image):
        """
        Step 3: Baseline Detection
        As per paper: "baseline of the word is determined by finding the row 
        with the maximum number of text pixels"
        
        Args:
            binary_image: Binarized image
            
        Returns:
            Baseline y-coordinate
        """
        print("\n" + "="*60)
        print("STEP 3: BASELINE DETECTION")
        print("="*60)
        
        # Calculate horizontal projection (sum of pixels in each row)
        horizontal_projection = np.sum(binary_image, axis=1)
        
        # Find row with maximum text pixels
        baseline_y = np.argmax(horizontal_projection)
        max_pixels = horizontal_projection[baseline_y]
        
        print(f"  - Baseline detected at row: {baseline_y}")
        print(f"  - Maximum pixel count at baseline: {max_pixels}")
        
        return baseline_y
    
    # =========================================================================
    # STEP 4: SEPARATION OF PRIMARY AND SECONDARY COMPONENTS
    # =========================================================================
    def separate_primary_secondary(self, components, baseline_y, binary_image):
        """
        Step 4: Separate Primary Ligatures and Diacritics (Secondary)
        As per paper: "Diacritics are differentiated from ligatures through 
        their position with respect to baseline"
        
        Primary components: Main body (touch or cross baseline)
        Secondary components: Diacritics (dots above/below baseline)
        
        Args:
            components: List of extracted components
            baseline_y: Detected baseline y-coordinate
            binary_image: Original binary image
            
        Returns:
            primary_components, secondary_components
        """
        print("\n" + "="*60)
        print("STEP 4: SEPARATION OF PRIMARY AND SECONDARY COMPONENTS")
        print("="*60)
        
        primary_components = []
        secondary_components = []
        
        # Calculate average component height for threshold
        if components:
            avg_height = np.mean([c['height'] for c in components])
            avg_area = np.mean([c['area'] for c in components])
        else:
            avg_height = 20
            avg_area = 100
        
        for comp in components:
            y_top = comp['y']
            y_bottom = comp['y'] + comp['height']
            
            # Check if component touches or crosses baseline
            touches_baseline = y_top <= baseline_y <= y_bottom
            
            # Components that are small and don't touch baseline are diacritics
            is_small = comp['area'] < avg_area * 0.3 or comp['height'] < avg_height * 0.5
            
            if touches_baseline or (not is_small and comp['height'] > avg_height * 0.4):
                comp['type'] = 'primary'
                comp['position'] = 'baseline'
                primary_components.append(comp)
            else:
                comp['type'] = 'secondary'
                # Determine position relative to baseline
                if comp['centroid_y'] < baseline_y:
                    comp['position'] = 'above'
                else:
                    comp['position'] = 'below'
                secondary_components.append(comp)
        
        print(f"  - Primary components (main body): {len(primary_components)}")
        print(f"  - Secondary components (diacritics): {len(secondary_components)}")
        
        # Count diacritics above and below
        above = sum(1 for c in secondary_components if c['position'] == 'above')
        below = sum(1 for c in secondary_components if c['position'] == 'below')
        print(f"    - Diacritics above baseline: {above}")
        print(f"    - Diacritics below baseline: {below}")
        
        return primary_components, secondary_components
    
    # =========================================================================
    # STEP 5: FEATURE EXTRACTION
    # =========================================================================
    def extract_features(self, component_image):
        """
        Step 5: Feature Extraction
        As per paper: "horizontal projection, vertical projection, 
        upper profile and lower profile"
        
        Args:
            component_image: Binary image of component
            
        Returns:
            Dictionary of features
        """
        height, width = component_image.shape
        
        features = {}
        
        # a) Horizontal Projection
        # "sum of pixel values in each row, normalized by width"
        horizontal_proj = np.sum(component_image, axis=1) / (width * 255)
        features['horizontal_projection'] = horizontal_proj
        
        # b) Vertical Projection  
        # "sum of pixel values in each column, normalized by height"
        vertical_proj = np.sum(component_image, axis=0) / (height * 255)
        features['vertical_projection'] = vertical_proj
        
        # c) Upper Profile
        # "distance of first text pixel from top, normalized by height"
        upper_profile = np.zeros(width)
        for col in range(width):
            column = component_image[:, col]
            text_pixels = np.where(column > 0)[0]
            if len(text_pixels) > 0:
                upper_profile[col] = text_pixels[0] / height
            else:
                upper_profile[col] = 1.0
        features['upper_profile'] = upper_profile
        
        # d) Lower Profile
        # "distance of last text pixel from top, normalized by height"
        lower_profile = np.zeros(width)
        for col in range(width):
            column = component_image[:, col]
            text_pixels = np.where(column > 0)[0]
            if len(text_pixels) > 0:
                lower_profile[col] = text_pixels[-1] / height
            else:
                lower_profile[col] = 0.0
        features['lower_profile'] = lower_profile
        
        return features
    
    # =========================================================================
    # STEP 6: SAVE COMPONENTS AS SEPARATE FILES
    # =========================================================================
    def save_components(self, components, original_image, prefix=""):
        """
        Step 6: Save each detected component as a separate file
        
        Args:
            components: List of components to save
            original_image: Original image for reference
            prefix: Prefix for filename
            
        Returns:
            List of saved file paths
        """
        print("\n" + "="*60)
        print("STEP 6: SAVING COMPONENTS AS SEPARATE FILES")
        print("="*60)
        
        saved_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Sort components right-to-left (Urdu reading order)
        sorted_components = sorted(components, key=lambda c: -c['x'])
        
        for idx, comp in enumerate(sorted_components):
            # Determine folder based on type
            if comp['type'] == 'primary':
                folder = self.primary_folder
                type_prefix = "primary"
            else:
                folder = self.secondary_folder
                type_prefix = f"diacritic_{comp['position']}"
            
            # Create filename
            filename = f"{prefix}{type_prefix}_{idx+1:04d}_{timestamp}.png"
            filepath = os.path.join(folder, filename)
            
            # Save binary component image
            cv2.imwrite(filepath, comp['image'])
            saved_files.append(filepath)
            
            # Also save to all_components folder
            all_filepath = os.path.join(self.all_components_folder, 
                                        f"comp_{idx+1:04d}_{timestamp}.png")
            cv2.imwrite(all_filepath, comp['image'])
            
            # Extract and save features
            features = self.extract_features(comp['image'])
            comp['features'] = features
        
        print(f"  - Saved {len(saved_files)} component images")
        print(f"  - Primary ligatures saved to: {self.primary_folder}")
        print(f"  - Diacritics saved to: {self.secondary_folder}")
        print(f"  - All components saved to: {self.all_components_folder}")
        
        return saved_files
    
    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    def create_visualization(self, original_image, components, baseline_y):
        """
        Create visualization showing detected components
        
        Args:
            original_image: Original image
            components: List of all components
            baseline_y: Detected baseline
            
        Returns:
            Visualization image
        """
        # Convert to color if grayscale
        if len(original_image.shape) == 2:
            vis_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        else:
            vis_image = original_image.copy()
        
        # Draw baseline (blue line)
        cv2.line(vis_image, (0, baseline_y), (vis_image.shape[1], baseline_y),
                 (255, 0, 0), 2)
        cv2.putText(vis_image, "Baseline", (10, baseline_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Draw bounding boxes
        for idx, comp in enumerate(components):
            x, y, w, h = comp['bbox']
            
            if comp['type'] == 'primary':
                color = (0, 255, 0)  # Green for primary
                label = f"P{idx+1}"
            else:
                if comp['position'] == 'above':
                    color = (0, 165, 255)  # Orange for above
                else:
                    color = (0, 255, 255)  # Yellow for below
                label = f"D{idx+1}"
            
            # Draw rectangle
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            cv2.putText(vis_image, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return vis_image
    
    # =========================================================================
    # MAIN DETECTION PIPELINE
    # =========================================================================
    def detect_and_extract(self, image_path):
        """
        Main function: Complete character detection and extraction pipeline
        Following all steps from the paper
        
        Args:
            image_path: Path to input image
            
        Returns:
            Tuple of (saved_files, visualization_image, components)
        """
        print("\n" + "#"*60)
        print("URDU OCR DETECTION SYSTEM")
        print("Based on: Shabbir & Siddiqi, IJACSA 2016")
        print("#"*60)
        
        # Load image
        print(f"\nLoading image: {image_path}")
        original_image = cv2.imread(image_path)
        
        if original_image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        print(f"Image dimensions: {original_image.shape[1]}x{original_image.shape[0]}")
        
        # Step 1: Preprocessing - Binarization
        binary_image = self.preprocess_binarization(original_image)
        
        # Step 2: Extract Ligatures using Connected Component Labeling
        components = self.extract_ligatures_connected_components(binary_image)
        
        if len(components) == 0:
            print("\nNo components found in image!")
            return [], original_image, []
        
        # Step 3: Baseline Detection
        baseline_y = self.detect_baseline(binary_image)
        
        # Step 4: Separate Primary and Secondary Components
        primary, secondary = self.separate_primary_secondary(
            components, baseline_y, binary_image
        )
        
        # Combine all components with their classification
        all_components = primary + secondary
        
        # Step 5 & 6: Extract Features and Save Components
        saved_files = self.save_components(all_components, original_image)
        
        # Create visualization
        visualization = self.create_visualization(
            original_image, all_components, baseline_y
        )
        
        # Summary
        print("\n" + "#"*60)
        print("DETECTION COMPLETE!")
        print("#"*60)
        print(f"Total components detected: {len(all_components)}")
        print(f"  - Primary ligatures: {len(primary)}")
        print(f"  - Diacritics: {len(secondary)}")
        print(f"Files saved to: {self.output_folder}")
        
        return saved_files, visualization, all_components


def main():
    """
    Main function to run the Urdu OCR detector
    """
    import sys
    
    print("="*60)
    print("URDU CHARACTER DETECTION SYSTEM")
    print("Based on paper: 'Optical Character Recognition System")
    print("for Urdu Words in Nastaliq Font'")
    print("By Safia Shabbir and Imran Siddiqi, Bahria University")
    print("="*60)
    
    # Get image path
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Default test image
        image_path = "test_urdu.png"
        print(f"\nUsage: python urdu_ocr_detector.py <image_path>")
        print(f"Using default: {image_path}\n")
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"\nError: Image file '{image_path}' not found!")
        print("\nPlease provide a valid image path.")
        print("Example: python urdu_ocr_detector.py urdu_text.png")
        
        # Create a sample test image for demonstration
        print("\nCreating a sample test image for demonstration...")
        create_sample_image()
        image_path = "test_urdu_sample.png"
    
    # Create detector
    detector = UrduOCRDetector(output_folder="extracted_characters")
    
    try:
        # Run detection
        saved_files, visualization, components = detector.detect_and_extract(image_path)
        
        # Save visualization
        viz_path = "detection_visualization.png"
        cv2.imwrite(viz_path, visualization)
        print(f"\nVisualization saved to: {viz_path}")
        
        # Save binary image
        binary_path = "binary_image.png"
        binary = detector.preprocess_binarization(cv2.imread(image_path))
        cv2.imwrite(binary_path, binary)
        print(f"Binary image saved to: {binary_path}")
        
        # Try to display
        try:
            cv2.imshow("Detection Results", visualization)
            print("\nPress any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            print("(Display not available - running in headless mode)")
        
        # Print saved files
        print("\n" + "-"*40)
        print("SAVED CHARACTER FILES:")
        print("-"*40)
        for f in saved_files[:15]:
            print(f"  {f}")
        if len(saved_files) > 15:
            print(f"  ... and {len(saved_files) - 15} more files")
            
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


def create_sample_image():
    """
    Create a sample test image with Urdu-like characters for demonstration
    """
    # Create a white image
    img = np.ones((200, 400, 3), dtype=np.uint8) * 255
    
    # Draw some sample shapes resembling Urdu characters
    # Main body (primary ligatures)
    cv2.ellipse(img, (100, 100), (30, 40), 0, 0, 360, (0, 0, 0), 2)
    cv2.line(img, (130, 100), (180, 100), (0, 0, 0), 3)
    cv2.ellipse(img, (200, 100), (20, 30), 0, 0, 360, (0, 0, 0), 2)
    cv2.line(img, (220, 100), (280, 80), (0, 0, 0), 3)
    cv2.ellipse(img, (300, 100), (25, 35), 0, 0, 360, (0, 0, 0), 2)
    
    # Diacritics (dots above and below)
    cv2.circle(img, (100, 50), 5, (0, 0, 0), -1)  # Dot above
    cv2.circle(img, (200, 150), 5, (0, 0, 0), -1)  # Dot below
    cv2.circle(img, (300, 45), 4, (0, 0, 0), -1)  # Two dots above
    cv2.circle(img, (310, 45), 4, (0, 0, 0), -1)
    
    cv2.imwrite("test_urdu_sample.png", img)
    print("Sample image created: test_urdu_sample.png")


if __name__ == "__main__":
    main()
