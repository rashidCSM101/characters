"""
Ligature Dictionary Builder
============================
Automatically processes images from input folder, extracts ligatures,
and builds a dictionary with duplicate handling

Usage: python build_ligature_dictionary.py
"""

import cv2
import numpy as np
import os
import json
import hashlib
from collections import defaultdict
from datetime import datetime


class LigatureDictionaryBuilder:
    """Builds a dictionary of unique ligatures by processing extracted characters"""
    
    def __init__(self, results_folder=None, output_folder="ligature_dictionary"):
        self.results_folder = results_folder
        self.output_folder = output_folder
        self.dict_folder = os.path.join(output_folder, "unique_ligatures")
        self.duplicates_folder = os.path.join(output_folder, "duplicates")
        
        # Create folders
        os.makedirs(self.dict_folder, exist_ok=True)
        os.makedirs(self.duplicates_folder, exist_ok=True)
        
        # Dictionary data
        self.ligature_dict = {}  # hash -> ligature_info
        self.hash_to_id = {}     # hash -> unique_id
        self.next_id = 1
        self.duplicate_count = 0
        
        # Statistics
        self.total_results_processed = 0
        self.total_chars_extracted = 0
        
    def compute_image_hash(self, img):
        """Compute perceptual hash for image comparison"""
        # Resize to standard size
        resized = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
        
        # Simple hash using pixel values
        hash_str = hashlib.md5(resized.tobytes()).hexdigest()
        return hash_str
    
    def are_images_similar(self, img1, img2, threshold=0.95):
        """Check if two images are similar using correlation"""
        # Resize both to same size
        size = (32, 32)
        img1_resized = cv2.resize(img1, size, interpolation=cv2.INTER_AREA)
        img2_resized = cv2.resize(img2, size, interpolation=cv2.INTER_AREA)
        
        # Normalize
        img1_norm = img1_resized.astype(float) / 255.0
        img2_norm = img2_resized.astype(float) / 255.0
        
        # Compute correlation
        correlation = np.corrcoef(img1_norm.flatten(), img2_norm.flatten())[0, 1]
        
        return correlation >= threshold
    
    def find_similar_ligature(self, img):
        """Find if similar ligature exists in dictionary"""
        for lig_hash, lig_info in self.ligature_dict.items():
            stored_img = cv2.imread(lig_info['path'], cv2.IMREAD_GRAYSCALE)
            if stored_img is not None:
                if self.are_images_similar(img, stored_img):
                    return lig_hash, lig_info
        return None, None
    
    def add_ligature(self, img, source_line, source_char_idx, diacritic_count):
        """Add ligature to dictionary (or update if duplicate)"""
        # Check for similar ligature
        similar_hash, similar_info = self.find_similar_ligature(img)
        
        if similar_hash:
            # Duplicate found
            self.duplicate_count += 1
            lig_id = similar_info['id']
            
            # Update occurrence count
            self.ligature_dict[similar_hash]['occurrences'] += 1
            self.ligature_dict[similar_hash]['sources'].append({
                'line': source_line,
                'char_idx': source_char_idx
            })
            
            # Save duplicate with reference
            dup_filename = f"dup_{self.duplicate_count:04d}_ref_{lig_id:04d}.png"
            dup_path = os.path.join(self.duplicates_folder, dup_filename)
            cv2.imwrite(dup_path, img)
            
            return lig_id, True  # True = is_duplicate
        else:
            # New unique ligature
            img_hash = self.compute_image_hash(img)
            lig_id = self.next_id
            self.next_id += 1
            
            # Save unique ligature
            filename = f"ligature_{lig_id:04d}_dots_{diacritic_count}.png"
            filepath = os.path.join(self.dict_folder, filename)
            cv2.imwrite(filepath, img)
            
            # Store in dictionary
            self.ligature_dict[img_hash] = {
                'id': lig_id,
                'path': filepath,
                'filename': filename,
                'diacritic_count': diacritic_count,
                'occurrences': 1,
                'sources': [{
                    'line': source_line,
                    'char_idx': source_char_idx
                }],
                'dimensions': {
                    'width': img.shape[1],
                    'height': img.shape[0]
                }
            }
            
            self.hash_to_id[img_hash] = lig_id
            
            return lig_id, False  # False = is_new
    
    def find_all_results_folders(self):
        """Find all results_* folders in current directory"""
        results_folders = []
        for item in os.listdir('.'):
            if os.path.isdir(item) and item.startswith('results_'):
                chars_folder = os.path.join(item, '4_extracted_chars')
                if os.path.exists(chars_folder):
                    results_folders.append(item)
        return sorted(results_folders)
    
    def process_results_folder(self, results_folder):
        """Process extracted characters from a results folder"""
        chars_folder = os.path.join(results_folder, '4_extracted_chars')
        
        print(f"\n📂 Processing: {results_folder}")
        
        if not os.path.exists(chars_folder):
            print(f"  ❌ Characters folder not found: {chars_folder}")
            return 0
        
        # Find all line folders
        line_folders = [
            f for f in os.listdir(chars_folder)
            if os.path.isdir(os.path.join(chars_folder, f)) and f.startswith('line_')
        ]
        
        if not line_folders:
            print(f"  ❌ No line folders found")
            return 0
        
        line_folders.sort()
        print(f"  📏 Found {len(line_folders)} line folders")
        
        chars_count = 0
        
        # Process each line folder
        for line_folder in line_folders:
            line_path = os.path.join(chars_folder, line_folder)
            line_num = line_folder.replace('line_', '')
            
            # Get all character images
            char_files = [f for f in os.listdir(line_path) if f.endswith('.png')]
            char_files.sort()
            
            print(f"    📄 {line_folder}: {len(char_files)} characters")
            
            for char_file in char_files:
                # Extract info from filename
                parts = char_file.replace('.png', '').split('_')
                diacritic_count = 0
                char_idx = 0
                
                try:
                    for i, part in enumerate(parts):
                        if part == 'char' and i+1 < len(parts):
                            char_idx = int(parts[i+1])
                        elif part == 'dots' and i+1 < len(parts):
                            diacritic_count = int(parts[i+1])
                except:
                    pass
                
                # Load image
                img_path = os.path.join(line_path, char_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    continue
                
                # Add to dictionary
                lig_id, is_duplicate = self.add_ligature(
                    img,
                    source_line=f"{results_folder}_L{line_num}",
                    source_char_idx=char_idx,
                    diacritic_count=diacritic_count
                )
                
                chars_count += 1
                status = "✓" if not is_duplicate else "≈"
                
        self.total_results_processed += 1
        self.total_chars_extracted += chars_count
        
        print(f"  ✅ Processed {chars_count} characters")
        
        return chars_count
    
    def build_from_extracted_chars(self):
        """Build dictionary from extracted characters in results folders"""
        print("="*70)
        print("AUTOMATIC LIGATURE DICTIONARY BUILDER")
        print("="*70)
        
        # Determine which results folders to process
        if self.results_folder:
            # Process specific folder
            if not os.path.exists(self.results_folder):
                print(f"\n❌ Error: Results folder not found: {self.results_folder}")
                return None
            results_folders = [self.results_folder]
        else:
            # Find all results folders
            results_folders = self.find_all_results_folders()
            
            if not results_folders:
                print("\n❌ No results_* folders found!")
                print("Please run complete_urdu_ocr.py first to generate extracted characters")
                return None
        
        print(f"\n📊 Found {len(results_folders)} results folder(s) to process:")
        for folder in results_folders:
            print(f"  • {folder}")
        
        # Process each results folder
        for results_folder in results_folders:
            self.process_results_folder(results_folder)
        
        # Summary
        print("\n" + "="*70)
        print("DICTIONARY BUILDING COMPLETE")
        print("="*70)
        print(f"\n📊 Final Statistics:")
        print(f"  Results folders processed: {self.total_results_processed}")
        print(f"  Total characters extracted: {self.total_chars_extracted}")
        print(f"  Unique ligatures: {len(self.ligature_dict)}")
        print(f"  Duplicates: {self.duplicate_count}")
        
        if self.total_chars_extracted > 0:
            compression = (len(self.ligature_dict) / self.total_chars_extracted * 100)
            print(f"  Compression ratio: {compression:.1f}%")
        
        # Save dictionary metadata
        self.save_dictionary_info()
        
        return {
            'results_processed': self.total_results_processed,
            'total_chars': self.total_chars_extracted,
            'unique_ligatures': len(self.ligature_dict),
            'duplicates': self.duplicate_count
        }
    
    def save_dictionary_info(self):
        """Save dictionary information to JSON"""
        # Prepare data for JSON
        dict_data = {
            'total_unique_ligatures': len(self.ligature_dict),
            'total_duplicates': self.duplicate_count,
            'ligatures': []
        }
        
        # Sort by ID
        sorted_ligatures = sorted(
            self.ligature_dict.items(),
            key=lambda x: x[1]['id']
        )
        
        for img_hash, lig_info in sorted_ligatures:
            dict_data['ligatures'].append({
                'id': lig_info['id'],
                'filename': lig_info['filename'],
                'diacritic_count': lig_info['diacritic_count'],
                'occurrences': lig_info['occurrences'],
                'dimensions': lig_info['dimensions'],
                'sources': lig_info['sources']
            })
        
        # Save JSON
        json_path = os.path.join(self.dict_folder, 'dictionary_info.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(dict_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Dictionary info saved: {json_path}")
        
        # Create summary file
        summary_path = os.path.join(self.dict_folder, 'summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("LIGATURE DICTIONARY SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Total Unique Ligatures: {len(self.ligature_dict)}\n")
            f.write(f"Total Duplicates: {self.duplicate_count}\n")
            f.write(f"Total Occurrences: {sum(l['occurrences'] for l in self.ligature_dict.values())}\n\n")
            
            f.write("Ligature Details:\n")
            f.write("-"*70 + "\n")
            
            for img_hash, lig_info in sorted_ligatures:
                f.write(f"\nID: {lig_info['id']:04d}\n")
                f.write(f"  File: {lig_info['filename']}\n")
                f.write(f"  Diacritics: {lig_info['diacritic_count']}\n")
                f.write(f"  Occurrences: {lig_info['occurrences']}\n")
                f.write(f"  Size: {lig_info['dimensions']['width']}x{lig_info['dimensions']['height']}\n")
                f.write(f"  Found in lines: {', '.join(str(s['line']) for s in lig_info['sources'])}\n")
        
        print(f"✅ Summary saved: {summary_path}")
        
        print(f"\n📁 Dictionary structure:")
        print(f"  {self.dict_folder}/")
        print(f"  ├── ligature_0001_dots_2.png")
        print(f"  ├── ligature_0002_dots_0.png")
        print(f"  ├── ...")
        print(f"  ├── duplicates/")
        print(f"  │   ├── dup_0001_ref_0003.png")
        print(f"  │   └── ...")
        print(f"  ├── dictionary_info.json")
        print(f"  └── summary.txt")
    
    def create_visual_dictionary(self):
        """Create a visual grid showing all unique ligatures"""
        if not self.ligature_dict:
            return
        
        print("\n📊 Creating visual dictionary...")
        
        # Parameters
        cell_size = 64
        padding = 10
        cols = 10
        
        sorted_ligatures = sorted(
            self.ligature_dict.items(),
            key=lambda x: x[1]['id']
        )
        
        num_ligatures = len(sorted_ligatures)
        rows = (num_ligatures + cols - 1) // cols
        
        # Create grid
        grid_width = cols * (cell_size + padding) + padding
        grid_height = rows * (cell_size + padding) + padding + 50  # Extra for title
        
        grid = np.ones((grid_height, grid_width), dtype=np.uint8) * 255
        
        # Add title
        cv2.putText(grid, f"Ligature Dictionary ({num_ligatures} unique)", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
        
        # Add ligatures
        for idx, (img_hash, lig_info) in enumerate(sorted_ligatures):
            row = idx // cols
            col = idx % cols
            
            x = col * (cell_size + padding) + padding
            y = row * (cell_size + padding) + padding + 50
            
            # Load and resize ligature
            img = cv2.imread(lig_info['path'], cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            # Resize maintaining aspect ratio
            h, w = img.shape
            scale = min(cell_size / w, cell_size / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Center in cell
            offset_x = (cell_size - new_w) // 2
            offset_y = (cell_size - new_h) // 2
            
            # Paste into grid
            grid[y+offset_y:y+offset_y+new_h, x+offset_x:x+offset_x+new_w] = resized
            
            # Draw border
            cv2.rectangle(grid, (x, y), (x+cell_size, y+cell_size), 0, 1)
            
            # Add ID label
            label = f"{lig_info['id']}"
            cv2.putText(grid, label, (x+2, y+cell_size-2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, 0, 1)
        
        # Save
        visual_path = os.path.join(self.dict_folder, 'visual_dictionary.png')
        cv2.imwrite(visual_path, grid)
        
        print(f"✅ Visual dictionary created: {visual_path}")


def main():
    import sys
    
    print("="*70)
    print("AUTOMATIC LIGATURE DICTIONARY BUILDER")
    print("="*70)
    print(f"\n🎯 What it does:")
    print("  1. Reads extracted characters from results_*/4_extracted_chars/")
    print("  2. Identifies unique ligatures")
    print("  3. Detects and handles duplicates")
    print("  4. Creates organized dictionary")
    print("  5. Generates visual grid")
    
    # Get results folder from argument (optional)
    results_folder = sys.argv[1] if len(sys.argv) > 1 else None
    output_folder = sys.argv[2] if len(sys.argv) > 2 else "ligature_dictionary"
    
    if results_folder:
        print(f"\n📂 Target folder: {results_folder}")
    else:
        print(f"\n📂 Mode: Auto-detect all results_* folders")
    
    # Build dictionary
    builder = LigatureDictionaryBuilder(results_folder, output_folder)
    stats = builder.build_from_extracted_chars()
    
    # Create visual dictionary
    if stats and stats['unique_ligatures'] > 0:
        builder.create_visual_dictionary()
    
    print("\n" + "="*70)
    print("✅ DICTIONARY BUILDING COMPLETE!")
    print("="*70)
    print(f"\n📁 Check output folder: {output_folder}/")
    print("  ├── unique_ligatures/ (all unique ligatures)")
    print("  ├── duplicates/ (duplicate instances)")
    print("  ├── dictionary_info.json")
    print("  ├── summary.txt")
    print("  └── visual_dictionary.png")
    
    print("\n📝 Usage examples:")
    print("  python build_ligature_dictionary.py              # Process all results_* folders")
    print("  python build_ligature_dictionary.py results_img3 # Process specific folder")


if __name__ == "__main__":
    main()
