"""
Batch Urdu OCR System - Process Multiple Images (ENHANCED)
===========================================================
Based on paper: "Optical Character Recognition System for Urdu Words in Nastaliq Font"
By Safia Shabbir and Imran Siddiqi, Bahria University

ENHANCED PREPROCESSING:
1. Contrast Enhancement (CLAHE)
2. Adaptive Thresholding
3. Noise Removal (Morphological Operations)
4. Skew Correction

Features:
- Characters detect karta hai
- Primary ligatures aur diacritics alag karta hai
- Binary images alag folder mein save karta hai
- Visualization images alag folder mein save karta hai
- Preprocessing steps bhi save karta hai
"""

import cv2
import numpy as np
import os
from datetime import datetime


class BatchUrduOCR:
    """
    Enhanced Batch processing for Urdu character detection
    """
    
    def __init__(self, input_folder="input_images", output_folder="output"):
        self.input_folder = input_folder
        self.output_folder = output_folder
        
        # Output folders
        self.binary_folder = os.path.join(output_folder, "binary_images")
        self.visualization_folder = os.path.join(output_folder, "visualization_images")
        self.characters_folder = os.path.join(output_folder, "extracted_characters")
        self.preprocessing_folder = os.path.join(output_folder, "preprocessing_steps")
        
        # Create folders
        folders = [
            self.input_folder,
            self.output_folder,
            self.binary_folder,
            self.visualization_folder,
            self.characters_folder,
            self.preprocessing_folder
        ]
        
        for folder in folders:
            if not os.path.exists(folder):
                os.makedirs(folder)
                print(f"✓ Created: {folder}")
        
        self.min_component_area = 3  # IMPROVED: Reduced from 10 to 3 to detect smaller characters
        self.supported_formats = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
        
        # IMPROVED: Multi-scale detection (disabled by default to avoid over-detection)
        self.use_multiscale = False  # Set to True if needed
        self.scales = [0.95, 1.0, 1.05]  # Narrower range to reduce duplicates
        
        # IMPROVED: Adaptive thresholding parameters (optimized single parameter)
        self.adaptive_block_size = 13  # Optimal for Urdu text
        self.adaptive_C = 3  # Optimal constant
    
    # =========================================================================
    # ENHANCEMENT 1: CONTRAST IMPROVEMENT (CLAHE)
    # =========================================================================
    def enhance_contrast(self, gray_image):
        """
        Contrast Limited Adaptive Histogram Equalization (CLAHE)
        Improves contrast in different regions of the image
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray_image)
        return enhanced
    
    # =========================================================================
    # ENHANCEMENT: BILATERAL FILTER DENOISING
    # =========================================================================
    def bilateral_denoise(self, gray_image):
        """
        Bilateral filter for edge-preserving denoising
        Better than Gaussian blur for text
        """
        denoised = cv2.bilateralFilter(gray_image, 9, 75, 75)
        return denoised
    
    # =========================================================================
    # ENHANCEMENT 2: SKEW CORRECTION
    # =========================================================================
    def detect_skew_angle(self, binary_image):
        """
        Detect skew angle using Hough Line Transform
        Returns angle in degrees
        """
        # Find edges
        edges = cv2.Canny(binary_image, 50, 150, apertureSize=3)
        
        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, 
            threshold=100, 
            minLineLength=100, 
            maxLineGap=10
        )
        
        if lines is None:
            return 0.0
        
        # Calculate angles
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:  # Avoid division by zero
                angle = np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi
                # Only consider small angles (likely text baseline)
                if -45 < angle < 45:
                    angles.append(angle)
        
        if not angles:
            return 0.0
        
        # Return median angle
        return np.median(angles)
    
    def correct_skew(self, image, angle):
        """
        Rotate image to correct skew
        """
        if abs(angle) < 0.5:  # Skip if angle is very small
            return image
        
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new image size to avoid cropping
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        
        # Adjust rotation matrix
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2
        
        # Rotate
        rotated = cv2.warpAffine(
            image, M, (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated
    
    # =========================================================================
    # ENHANCEMENT 3: ADAPTIVE THRESHOLDING (IMPROVED - MULTI-PARAMETER)
    # =========================================================================
    def adaptive_binarization(self, gray_image):
        """
        Adaptive thresholding for better results with varying lighting
        IMPROVED: Try multiple parameters and combine results
        """
        # Apply Gaussian Adaptive Thresholding
        binary = cv2.adaptiveThreshold(
            gray_image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=11,  # Size of neighborhood
            C=2            # Constant subtracted from mean
        )
        return binary
    
    def multi_parameter_binarization(self, gray_image):
        """
        IMPROVED: Try multiple adaptive thresholding parameters and combine
        This catches characters that might be missed with single parameters
        """
        all_binaries = []
        
        # Try different combinations of block size and C value
        for block_size in self.adaptive_block_sizes:
            for C in self.adaptive_C_values:
                binary = cv2.adaptiveThreshold(
                    gray_image,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV,
                    blockSize=block_size,
                    C=C
                )
                all_binaries.append(binary)
        
        # Combine all results using OR operation (union)
        combined = np.zeros_like(all_binaries[0])
        for binary in all_binaries:
            combined = cv2.bitwise_or(combined, binary)
        
        return combined
    
    # =========================================================================
    # ENHANCEMENT 4: NOISE REMOVAL (MORPHOLOGICAL OPERATIONS)
    # =========================================================================
    def remove_noise_morphology(self, binary_image):
        """
        Remove noise using morphological operations:
        - Opening: Removes small white noise (foreground)
        - Closing: Fills small holes in characters
        - IMPROVED: Gentle closing to avoid merging separate characters
        """
        # Define kernels - REDUCED size to avoid merging characters
        kernel_tiny = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        
        # Step 1: Opening - remove small noise particles
        opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel_tiny)
        
        # Step 2: Gentle Closing - fill only very small holes, avoid merging
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_small)
        
        return closed
    
    # =========================================================================
    # COMPLETE ENHANCED PREPROCESSING PIPELINE
    # =========================================================================
    def preprocess_binarization(self, image, save_steps=False, base_name=""):
        """
        Enhanced preprocessing pipeline:
        1. Convert to grayscale
        2. Contrast enhancement (CLAHE)
        3. Skew detection and correction
        4. Adaptive thresholding
        5. Noise removal (morphological operations)
        """
        steps = {}
        
        # Step 1: Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        steps['1_grayscale'] = gray
        
        # Step 1.5: IMPROVED - Bilateral filter denoising
        denoised = self.bilateral_denoise(gray)
        steps['1.5_denoised'] = denoised
        
        # Step 2: Contrast Enhancement (CLAHE)
        enhanced = self.enhance_contrast(denoised)
        steps['2_contrast_enhanced'] = enhanced
        
        # Step 3: Initial binarization for skew detection
        _, initial_binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Step 4: Skew Detection and Correction
        skew_angle = self.detect_skew_angle(initial_binary)
        if abs(skew_angle) > 0.5:
            enhanced = self.correct_skew(enhanced, skew_angle)
            steps['3_skew_corrected'] = enhanced
            print(f"    ↳ Skew corrected: {skew_angle:.2f}°")
        else:
            steps['3_skew_corrected'] = enhanced
        
        # Step 5: IMPROVED - Optimized Adaptive Thresholding
        # Using single optimized parameters instead of multi-parameter (reduces false positives)
        binary_adaptive = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, blockSize=13, C=3
        )
        steps['4_adaptive_threshold'] = binary_adaptive
        
        # Step 6: Noise Removal (Morphological Operations)
        binary_clean = self.remove_noise_morphology(binary_adaptive)
        steps['5_noise_removed'] = binary_clean
        
        # Save preprocessing steps if requested
        if save_steps and base_name:
            self.save_preprocessing_steps(steps, base_name)
        
        return binary_clean
    
    def save_preprocessing_steps(self, steps, base_name):
        """Save all preprocessing steps as images"""
        for step_name, step_image in steps.items():
            filepath = os.path.join(self.preprocessing_folder, f"{base_name}_{step_name}.png")
            cv2.imwrite(filepath, step_image)
    
    # =========================================================================
    # IMPROVEMENT: SPLIT MERGED COMPONENTS USING DISTANCE TRANSFORM
    # =========================================================================
    def split_merged_components(self, binary_component):
        """
        Split touching/merged characters using distance transform and watershed
        Returns list of separated components
        """
        if binary_component.sum() == 0:
            return [binary_component]
        
        # Distance transform
        dist = cv2.distanceTransform(binary_component, cv2.DIST_L2, 5)
        
        # Find peaks (character centers)
        ret, sure_fg = cv2.threshold(dist, 0.3 * dist.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        # Find markers
        ret, markers = cv2.connectedComponents(sure_fg)
        
        # If only one marker, no need to split
        if ret <= 2:
            return [binary_component]
        
        # Watershed segmentation
        # Convert to 3-channel for watershed
        component_3ch = cv2.cvtColor(binary_component, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(component_3ch, markers)
        
        # Extract individual components
        split_components = []
        for label in range(2, ret + 1):
            mask = np.where(markers == label, 255, 0).astype(np.uint8)
            if mask.sum() > 0:
                split_components.append(mask)
        
        return split_components if len(split_components) > 0 else [binary_component]
    
    # =========================================================================
    # IMPROVEMENT: REMOVE PAGE BORDERS
    # =========================================================================
    def remove_page_borders(self, binary_image):
        """
        Remove page borders and crop to content area
        """
        # Find contours
        contours, _ = cv2.findContours(binary_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return binary_image
        
        # Get bounding box of all content
        all_points = np.concatenate([cnt for cnt in contours])
        x, y, w, h = cv2.boundingRect(all_points)
        
        # Add small padding
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(binary_image.shape[1] - x, w + 2 * padding)
        h = min(binary_image.shape[0] - y, h + 2 * padding)
        
        # Crop
        cropped = binary_image[y:y+h, x:x+w]
        
        return cropped
    
    def extract_components(self, binary_image):
        """Step 2: Connected Component Labeling with IMPROVED statistical filtering"""
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_image, connectivity=8
        )
        
        # First pass: collect all components above minimum threshold
        all_components = []
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
                
                all_components.append({
                    'id': i, 'x': x, 'y': y, 'width': w, 'height': h,
                    'area': area, 'centroid_x': cx, 'centroid_y': cy,
                    'image': component_image, 'bbox': (x, y, w, h)
                })
        
        # IMPROVED: Statistical outlier filtering (remove only extreme outliers)
        if len(all_components) > 5:  # Only apply if we have enough components
            areas = [c['area'] for c in all_components]
            mean_area = np.mean(areas)
            std_area = np.std(areas)
            
            # Keep all small components (area < mean) and larger ones within reasonable bounds
            # This ensures we don't lose small diacritics
            components = [c for c in all_components 
                         if c['area'] <= mean_area or c['area'] < mean_area + 5 * std_area]
        else:
            components = all_components
        
        return components
    
    # =========================================================================
    # IMPROVEMENT: MULTI-SCALE DETECTION
    # =========================================================================
    def multiscale_detect_components(self, binary_image):
        """
        IMPROVED: Detect components at multiple scales and merge results
        This catches characters of different sizes
        """
        h, w = binary_image.shape
        all_components = []
        
        for scale in self.scales:
            # Resize image
            if scale != 1.0:
                new_w = int(w * scale)
                new_h = int(h * scale)
                scaled = cv2.resize(binary_image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            else:
                scaled = binary_image.copy()
            
            # Detect components at this scale
            components = self.extract_components(scaled)
            
            # Scale coordinates back to original size
            if scale != 1.0:
                for comp in components:
                    comp['x'] = int(comp['x'] / scale)
                    comp['y'] = int(comp['y'] / scale)
                    comp['width'] = int(comp['width'] / scale)
                    comp['height'] = int(comp['height'] / scale)
                    comp['centroid_x'] = comp['centroid_x'] / scale
                    comp['centroid_y'] = comp['centroid_y'] / scale
                    comp['bbox'] = (comp['x'], comp['y'], comp['width'], comp['height'])
                    # Resize component image back
                    if comp['image'] is not None:
                        comp['image'] = cv2.resize(comp['image'], 
                                                   (comp['width'], comp['height']), 
                                                   interpolation=cv2.INTER_CUBIC)
            
            all_components.extend(components)
        
        # Remove duplicate detections using Non-Maximum Suppression
        unique_components = self.non_maximum_suppression(all_components)
        
        return unique_components
    
    def non_maximum_suppression(self, components, iou_threshold=0.5):
        """
        Remove duplicate component detections using IoU-based NMS
        """
        if len(components) == 0:
            return []
        
        # Sort by area (larger first)
        components = sorted(components, key=lambda c: c['area'], reverse=True)
        
        keep = []
        
        while len(components) > 0:
            # Take the largest remaining component
            current = components.pop(0)
            keep.append(current)
            
            # Remove components that overlap significantly with current
            # IMPROVED: More aggressive overlap threshold for multi-scale
            remaining = []
            for comp in components:
                iou = self.calculate_iou(current['bbox'], comp['bbox'])
                if iou < 0.3:  # Reduced from 0.5 to 0.3 for stricter filtering
                    remaining.append(comp)
            
            components = remaining
        
        return keep
    
    def calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union (IoU) between two bounding boxes
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
    
    def detect_baseline(self, binary_image):
        """Step 3: Baseline Detection - IMPROVED with robust method"""
        # Method 1: Traditional horizontal projection
        horizontal_projection = np.sum(binary_image, axis=1)
        baseline_y_proj = np.argmax(horizontal_projection)
        
        # Method 2: IMPROVED - Contour-based (more robust)
        contours, _ = cv2.findContours(binary_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            # Get bottom-most points of all contours
            bottom_points = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                bottom_points.append(y + h)
            
            # Use median for robustness (less affected by outliers)
            baseline_y_contour = int(np.median(bottom_points))
            
            # Use average of both methods
            baseline_y = int((baseline_y_proj + baseline_y_contour) / 2)
        else:
            baseline_y = baseline_y_proj
        
        return baseline_y
    
    def separate_primary_secondary(self, components, baseline_y):
        """Step 4: Separate Primary and Secondary Components - IMPROVED with multiple criteria"""
        primary_components = []
        secondary_components = []
        
        if not components:
            return [], []
        
        avg_height = np.mean([c['height'] for c in components])
        avg_area = np.mean([c['area'] for c in components])
        avg_width = np.mean([c['width'] for c in components])
        
        for comp in components:
            y_top = comp['y']
            y_bottom = comp['y'] + comp['height']
            
            # Criterion 1: Baseline touching
            touches_baseline = y_top <= baseline_y <= y_bottom
            
            # Criterion 2: Size relative to average
            relative_area = comp['area'] / avg_area
            relative_height = comp['height'] / avg_height
            
            # Criterion 3: IMPROVED - Aspect ratio (diacritics tend to be circular)
            aspect_ratio = comp['width'] / comp['height'] if comp['height'] > 0 else 1
            is_round = 0.6 < aspect_ratio < 1.4
            
            # Criterion 4: IMPROVED - Distance from baseline
            distance_from_baseline = min(abs(y_top - baseline_y), abs(y_bottom - baseline_y))
            is_far_from_baseline = distance_from_baseline > avg_height * 0.3
            
            # IMPROVED CLASSIFICATION LOGIC:
            # Diacritics: small, round, far from baseline
            is_diacritic = (
                relative_area < 0.25 and 
                is_far_from_baseline and 
                is_round
            ) or (
                relative_height < 0.4 and 
                relative_area < 0.2
            )
            
            if not is_diacritic and (touches_baseline or relative_height > 0.45):
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
        """Process one image with ENHANCED preprocessing"""
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
        
        # Step 1: ENHANCED Binarization (with all preprocessing steps)
        print("  📌 ENHANCED PREPROCESSING:")
        print("    ↳ Contrast Enhancement (CLAHE)...")
        print("    ↳ Adaptive Thresholding...")
        print("    ↳ Noise Removal (Morphological Ops)...")
        binary_image = self.preprocess_binarization(original_image, save_steps=True, base_name=base_name)
        binary_path = os.path.join(self.binary_folder, f"{base_name}_binary.png")
        cv2.imwrite(binary_path, binary_image)
        print(f"  ✓ Binary saved: {base_name}_binary.png")
        print(f"  ✓ Preprocessing steps saved to: preprocessing_steps/")
        
        # Step 2: Extract components with MULTI-SCALE DETECTION
        if self.use_multiscale:
            print(f"  📌 Multi-scale detection at scales: {self.scales}")
            components = self.multiscale_detect_components(binary_image)
            print(f"  ✓ Components found (multi-scale): {len(components)}")
        else:
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
