"""
Feature Extraction Visualization
================================
Based on paper methodology - Figure 3 style visualization
Shows: Horizontal Projection, Vertical Projection, Upper Profile, Lower Profile
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def extract_and_visualize_features(image_path, output_path="features_visualization.png"):
    """
    Extract features from a character/ligature image and visualize them
    Similar to Figure 3 in the paper
    
    Args:
        image_path: Path to character image
        output_path: Path to save visualization
    """
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load {image_path}")
        return
    
    # Binarize if not already
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    height, width = binary.shape
    
    # =========================================
    # Feature Extraction (as per paper)
    # =========================================
    
    # a) Horizontal Projection - sum of pixels in each row
    horizontal_proj = np.sum(binary, axis=1) / 255
    horizontal_proj_normalized = horizontal_proj / width
    
    # b) Vertical Projection - sum of pixels in each column
    vertical_proj = np.sum(binary, axis=0) / 255
    vertical_proj_normalized = vertical_proj / height
    
    # c) Upper Profile - distance of first text pixel from top
    upper_profile = np.zeros(width)
    for col in range(width):
        column = binary[:, col]
        text_pixels = np.where(column > 0)[0]
        if len(text_pixels) > 0:
            upper_profile[col] = text_pixels[0]
        else:
            upper_profile[col] = height
    upper_profile_normalized = upper_profile / height
    
    # d) Lower Profile - distance of last text pixel from top
    lower_profile = np.zeros(width)
    for col in range(width):
        column = binary[:, col]
        text_pixels = np.where(column > 0)[0]
        if len(text_pixels) > 0:
            lower_profile[col] = text_pixels[-1]
        else:
            lower_profile[col] = 0
    lower_profile_normalized = lower_profile / height
    
    # =========================================
    # Visualization (Figure 3 style)
    # =========================================
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Feature Extraction (Paper: Shabbir & Siddiqi, 2016)', fontsize=14)
    
    # Original image
    axes[0, 0].imshow(binary, cmap='gray')
    axes[0, 0].set_title('Original Ligature')
    axes[0, 0].axis('off')
    
    # Horizontal Projection
    axes[0, 1].barh(range(height), horizontal_proj_normalized, color='blue')
    axes[0, 1].set_title('Horizontal Projection')
    axes[0, 1].set_xlabel('Normalized Sum')
    axes[0, 1].set_ylabel('Row')
    axes[0, 1].invert_yaxis()
    
    # Vertical Projection
    axes[0, 2].bar(range(width), vertical_proj_normalized, color='green')
    axes[0, 2].set_title('Vertical Projection')
    axes[0, 2].set_xlabel('Column')
    axes[0, 2].set_ylabel('Normalized Sum')
    
    # Upper Profile
    axes[1, 0].plot(range(width), upper_profile_normalized, color='red', linewidth=2)
    axes[1, 0].fill_between(range(width), upper_profile_normalized, alpha=0.3, color='red')
    axes[1, 0].set_title('Upper Profile')
    axes[1, 0].set_xlabel('Column')
    axes[1, 0].set_ylabel('Normalized Distance')
    axes[1, 0].invert_yaxis()
    
    # Lower Profile
    axes[1, 1].plot(range(width), lower_profile_normalized, color='purple', linewidth=2)
    axes[1, 1].fill_between(range(width), lower_profile_normalized, alpha=0.3, color='purple')
    axes[1, 1].set_title('Lower Profile')
    axes[1, 1].set_xlabel('Column')
    axes[1, 1].set_ylabel('Normalized Distance')
    axes[1, 1].invert_yaxis()
    
    # Combined profiles overlay on image
    axes[1, 2].imshow(binary, cmap='gray', alpha=0.5)
    axes[1, 2].plot(range(width), upper_profile, 'r-', linewidth=2, label='Upper Profile')
    axes[1, 2].plot(range(width), lower_profile, 'b-', linewidth=2, label='Lower Profile')
    axes[1, 2].set_title('Profiles Overlay')
    axes[1, 2].legend()
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Feature visualization saved to: {output_path}")
    
    return {
        'horizontal_projection': horizontal_proj_normalized,
        'vertical_projection': vertical_proj_normalized,
        'upper_profile': upper_profile_normalized,
        'lower_profile': lower_profile_normalized
    }


def visualize_all_components(components_folder, output_folder="feature_visualizations"):
    """
    Visualize features for all extracted components
    
    Args:
        components_folder: Folder containing extracted component images
        output_folder: Folder to save visualizations
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get all PNG files
    image_files = [f for f in os.listdir(components_folder) if f.endswith('.png')]
    
    print(f"Found {len(image_files)} component images")
    
    for img_file in image_files:
        img_path = os.path.join(components_folder, img_file)
        output_path = os.path.join(output_folder, f"features_{img_file}")
        
        try:
            extract_and_visualize_features(img_path, output_path)
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
    
    print(f"\nAll visualizations saved to: {output_folder}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        extract_and_visualize_features(image_path)
    else:
        print("Usage: python feature_visualization.py <image_path>")
        print("\nOr visualize all components:")
        print("  python feature_visualization.py --all <components_folder>")
        
        if len(sys.argv) > 2 and sys.argv[1] == "--all":
            visualize_all_components(sys.argv[2])
