import os
import cv2
import yaml
import random
import requests
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict
from collections import defaultdict

from pathlib import Path
from tqdm.auto import tqdm
from collections import Counter

# Special libraries for Persian/Arabic text rendering in plots
from bidi.algorithm import get_display
from arabic_reshaper import reshape

def clean_noise(root_dir: str, ksize: int = 3):
    """
    This function traverses through a directory and its subdirectories to remove 
    salt-and-pepper noise from images using a Median Blur filter. It overwrites 
    the original images with their denoised versions.

    Parameters:
    root_dir (str): The path of the root directory containing the images to clean.
    ksize (int): The aperture linear size for the blur; must be an odd integer 
                 (e.g., 3, 5). Larger values remove more noise but blur the image.

    Example:
    clean_noise('/kaggle/working/balanced_alphabet_dataset/train/images', ksize=3)
    """
    # Supported image extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    print(f"Starting noise cleanup in: {root_dir}")
    
    try:
        count = 0
        for subdir, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    input_path = os.path.join(subdir, file)
                    
                    # Load the image
                    img = cv2.imread(input_path)
                    
                    if img is None:
                        print(f"Skipping (could not read): {file}")
                        continue

                    # Apply Median Blur to remove noise
                    denoised_img = cv2.medianBlur(img, ksize)
                    
                    # Overwrite the original file with the denoised result
                    cv2.imwrite(input_path, denoised_img)
                    count += 1
        
        print(f"Cleanup complete. Processed {count} images.")

    except Exception as e:
        print(f"Error during noise cleanup: {e}")

def read_yaml(file_path: str):
    """
    This function reads and parses a YAML file from the specified path.
    It uses SafeLoader for secure parsing and handles common file and 
    formatting errors.

    Parameters:
    file_path (str): The path to the YAML file to be read.

    Returns:
    dict: The contents of the YAML file as a Python dictionary if successful, 
          otherwise None.

    Example:
    config = read_yaml('/kaggle/working/data.yaml')
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # Use SafeLoader to avoid executing arbitrary code within the YAML file
            data = yaml.load(file, Loader=yaml.SafeLoader)
            return data
            
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file at {file_path}: {exc}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading {file_path}: {e}")
        return None

def scan_yolo_labels(dataset_root: str):
    """
    Scans YOLO label files, locates the YAML config for character mapping, 
    and displays a clean report of counts and percentages per split.
    The raw dictionary output is suppressed for a cleaner notebook experience.

    Parameters:
    dataset_root (str): Root directory containing the splits and YAML file.

    Example:
    scan_yolo_labels('/kaggle/working/balanced_alphabet_dataset')

    Notes:
    - Automatically detects 'train', 'val', 'valid', and 'test' folders.
    - Uses 'arabic_reshaper' and 'python-bidi' for correct Persian text rendering.
    - Only displays mapped character names; raw IDs are hidden if YAML is found.
    """
    root = Path(dataset_root)
    # Checking common split names
    splits = ["train", "val", "valid", "test"]
    
    # 1. Automatically find the YAML file
    yaml_files = list(root.glob("*.yaml")) + list(root.glob("*.yml"))
    class_names = []
    
    if yaml_files:
        try:
            with open(yaml_files[0], 'r', encoding='utf-8') as f:
                data_config = yaml.safe_load(f)
                class_names = data_config.get('names', [])
        except Exception as e:
            print(f"Warning: Could not parse YAML: {e}")

    def get_label(cid):
        if cid < len(class_names):
            # Reshape and handle BiDi for correct Persian display
            return get_display(reshape(class_names[cid]))
        return f"ID {cid}"

    split_results = []
    grand_total_labels = 0

    # 2. Collect statistics
    for sp in splits:
        label_dir = root / sp / "labels"
        if not label_dir.exists():
            continue
            
        char_counts = Counter()
        file_list = list(label_dir.rglob("*.txt"))
        
        for p in file_list:
            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
                for line in content.splitlines():
                    line = line.strip()
                    if not line: continue
                    parts = line.split()
                    try:
                        cid = int(float(parts[0]))
                        char_counts[cid] += 1
                    except: continue
            except: continue
        
        if char_counts:
            split_sum = sum(char_counts.values())
            grand_total_labels += split_sum
            split_results.append((sp, char_counts, split_sum, len(file_list)))

    # 3. Display formatted report
    if grand_total_labels == 0:
        print("No labels found in the specified directory.")
        return

    print(f"\n{'#'*45}")
    print(f"{'DATASET DISTRIBUTION REPORT':^45}")
    print(f"{'#'*45}")

    for sp_name, counts, sp_total, file_count in split_results:
        sp_pct = (sp_total / grand_total_labels) * 100
        
        print(f"\n▶ SPLIT: {sp_name.upper()}")
        print(f"  Files: {file_count:,} | Total Labels: {sp_total:,} ({sp_pct:.2f}% of total)")
        print(f"  {'-'*40}")
        
        # Sort by character ID for consistent order
        for cid in sorted(counts.keys()):
            char_name = get_label(cid)
            count = counts[cid]
            pct = (count / sp_total) * 100
            print(f"  {char_name:<10} : {count:>6,} ({pct:>5.1f}%)")
    
    print(f"\n{'#'*45}")
    print(f"Total instances scanned: {grand_total_labels:,}")
    print(f"{'#'*45}\n")

def plot_dataset_distribution(dataset_root: str):
    """
    Dynamically discovers any subdirectories containing a 'labels' folder within 
    the dataset root and generates a grouped bar chart comparing class 
    distributions across these discovered splits.

    Parameters:
    dataset_root (str): The root directory to scan for YOLO-style splits.

    Example:
    plot_dataset_distribution('/kaggle/working/my_custom_dataset')
    
    Notes:
    - Automatically finds YAML in the root or one level up.
    - Scans for any folder containing a 'labels' sub-directory.
    - Displays instance counts vertically with clear spacing.
    """
    root = Path(dataset_root)
    
    # 1. Detect YAML and Class Names
    yaml_files = list(root.glob("*.yaml")) + list(root.glob("*.yml")) + list(root.parent.glob("*.yaml"))
    class_names = []
    if yaml_files:
        try:
            with open(yaml_files[0], 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                class_names = data.get('names', [])
                print(f"✅ Auto-detected YAML: {yaml_files[0].name}")
        except Exception as e:
            print(f"⚠️ Warning: Found YAML but failed to parse: {e}")

    # 2. Dynamic Split Discovery
    # We look for all 'labels' directories and treat their parent as the split name
    all_data = {} 
    label_dirs = list(root.rglob("labels"))
    
    if not label_dirs:
        print(f"❌ Error: No 'labels' subdirectories found in {dataset_root}.")
        return

    for ld in label_dirs:
        split_name = ld.parent.name
        counts = Counter()
        label_files = list(ld.glob("*.txt"))
        
        if not label_files:
            continue
            
        for p in label_files:
            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
                for line in content.splitlines():
                    parts = line.split()
                    if parts:
                        try:
                            cid = int(float(parts[0]))
                            counts[cid] += 1
                        except ValueError: continue
            except Exception: continue
        
        all_data[split_name] = counts
        print(f"🔎 Scanned Split [{split_name}]: found {len(label_files)} label files.")

    found_splits = sorted(all_data.keys())

    # 3. Prepare Data for Plotting
    found_ids = []
    for c in all_data.values():
        found_ids.extend(c.keys())
    
    max_id_found = max(found_ids) if found_ids else 0
    num_classes = max(len(class_names), max_id_found + 1)
    
    indices = np.arange(num_classes)
    labels = [get_display(reshape(str(class_names[i] if i < len(class_names) else f"ID {i}"))) for i in range(num_classes)]

    # 4. Visualization
    plt.figure(figsize=(25, 12))
    sns.set_style("whitegrid")
    
    width = 0.85 / len(found_splits)
    colors = sns.color_palette("muted", len(found_splits))
    
    max_val = max([max(c.values()) if c else 0 for c in all_data.values()])
    y_offset = max_val * 0.02

    for i, sp in enumerate(found_splits):
        counts = all_data[sp]
        values = [counts[j] for j in range(num_classes)]
        pos = indices - 0.4 + (i * width) + (width / 2)
        
        bars = plt.bar(pos, values, width, label=sp.upper(), color=colors[i], edgecolor='black', alpha=0.8)
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                plt.text(
                    bar.get_x() + bar.get_width()/2, 
                    height + y_offset, 
                    f'{int(height):,}', 
                    ha='center', va='bottom', fontsize=9, rotation=90, fontweight='bold'
                )

    plt.title(f'Multi-Split Class Distribution: {root.name}', fontsize=22, fontweight='bold', pad=40)
    plt.xlabel('Characters (Persian)', fontsize=16, fontweight='bold')
    plt.ylabel('Instance Count', fontsize=16, fontweight='bold')
    plt.xticks(indices, labels, rotation=0, fontsize=13)
    plt.legend(fontsize=14, title="Discovered Splits", title_fontsize='13', shadow=True)
    
    plt.ylim(0, max_val * 1.3) 
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    
def plot_random_images(root_dir: str, num_images: int = 5):
    """
    Selects random images from the dataset, overlays YOLO bounding boxes, 
    and displays them with labels. Handles automatic YAML 
    discovery and image-to-label path mapping.

    Parameters:
    root_dir (str): The directory to search for images.
    num_images (int): Number of random samples to display.

    Example:
    plot_random_images('/kaggle/working/dataset_path', num_images=4)
    """
    # 1. Automatic YAML Discovery (Search in root or parent)
    root_path = Path(root_dir)
    yaml_search = list(root_path.glob("*.yaml")) + list(root_path.parent.glob("*.yaml"))
    
    if not yaml_search:
        print("❌ Error: YAML configuration file not found.")
        return
        
    with open(yaml_search[0], 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    class_names = data.get('names', [])

    # 2. Collect Image Paths
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    image_paths = []
    for subdir, _, files in os.walk(root_dir):
        if 'images' in subdir.lower():
            for file in files:
                if file.lower().endswith(valid_exts):
                    image_paths.append(os.path.join(subdir, file))

    if not image_paths:
        print(f"❌ No images found in: {root_dir}")
        return

    # 3. Selection and Plotting Setup
    num_to_show = min(num_images, len(image_paths))
    random_images = random.sample(image_paths, num_to_show)
    
    fig, axes = plt.subplots(1, num_to_show, figsize=(22, 10))
    if num_to_show == 1: axes = [axes]

    for i, img_path in enumerate(random_images):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        ax = axes[i]

        # 4. Map Image to Label Path
        # Logic: find 'images' in path and replace with 'labels', change ext to .txt
        file_name = os.path.basename(img_path)
        img_name_no_ext = os.path.splitext(file_name)[0]
        
        # This handles your 'cleaned_' prefix replacement safely
        label_name = img_name_no_ext.replace('cleaned_', '') + '.txt'
        label_path = img_path.replace('/images/', '/labels/').replace(file_name, label_name)

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 5: continue
                    
                    cls_id = int(float(parts[0]))
                    x_c, y_c, bw, bh = map(float, parts[1:])
                    
                    # Convert YOLO normalized to Pixel coordinates
                    x1 = int((x_c - bw/2) * w)
                    y1 = int((y_c - bh/2) * h)
                    x2 = int((x_c + bw/2) * w)
                    y2 = int((y_c + bh/2) * h)

                    # Draw Bounding Box
                    cv2.rectangle(img, (x1, y1), (x2, y2), (50, 255, 50), 3)
                    
                    # 5. Persian Text Processing
                    if cls_id < len(class_names):
                        raw_name = class_names[cls_id]
                        display_name = get_display(reshape(str(raw_name)))
                    else:
                        display_name = str(cls_id)
                    
                    # Positioning text above the box
                    ax.text(int(x_c * w), y1 - 12, display_name, 
                            fontsize=14, color='white', fontweight='bold',
                            ha='center', va='bottom',
                            bbox=dict(facecolor='green', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))

        ax.imshow(img)
        ax.set_title(file_name, fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def download_yolo_model(url: str, save_path: str):
    """
    Downloads a YOLO model weights file from a URL with a progress bar.
    Skips the download if the file already exists at the destination.

    Parameters:
    url (str): The direct download link for the model (.pt or .onnx).
    save_path (str): Local path where the model should be saved.

    Example:
    download_yolo_model("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11m.pt", "models/yolo11m.pt")
    """
    save_path_obj = Path(save_path)
    
    # 1. Skip if already exists
    if save_path_obj.exists():
        print(f"✅ Model already exists at: {save_path}")
        return

    # 2. Prepare directory
    save_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # 3. Download with progress bar
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() # Check for 404/500 errors
        
        total_size = int(response.headers.get('content-length', 0))
        
        print(f"📥 Downloading YOLO model to {save_path}...")
        
        with open(save_path, 'wb') as file, tqdm(
            desc=save_path_obj.name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
                
        print(f"✨ Download complete: {save_path}")
        
    except Exception as e:
        if save_path_obj.exists():
            save_path_obj.unlink() # Delete partial download on failure
        print(f"❌ Failed to download model: {e}")

def save_dataset_yaml(config_dict, filename='data.yaml'):
    """
    Takes a dictionary of dataset information and saves it as a 
    YOLO-compliant YAML file with full Unicode support.

    Parameters:
    config_dict (dict): The dictionary containing 'path', 'train', 'val', 'nc', 'names', etc.
    filename (str): The name/path of the YAML file to be created. Defaults to 'data.yaml'.

    Example:
    >>> dataset_info = {
    ...     'path': '/kaggle/working/data',
    ...     'train': 'train/images',
    ...     'val': 'val/images',
    ...     'nc': 3,
    ...     'names': ['class_A', 'class_B', 'class_C']
    ... }
    >>> save_dataset_yaml(dataset_info)

    Notes:
    - Automatically creates the parent directory if it doesn't exist.
    - Uses 'allow_unicode=True' to ensure Persian/Arabic characters remain readable.
    - Sets 'sort_keys=False' to preserve the logical order of your configuration.
    """
    # 1. Determine the save path
    # If a path is provided in the dict, we use it as the base
    root_dir = config_dict.get('path', '.')
    save_path = Path(root_dir) / filename

    # 2. Ensure the parent directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 3. Write the YAML
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(
                config_dict, 
                f, 
                allow_unicode=True, 
                default_flow_style=False, 
                sort_keys=False
            )
        print(f"✅ YAML successfully created at: {save_path}")
    except Exception as e:
        print(f"❌ Error writing YAML: {e}")

def count_files(root_dir: Path, subsets: List[str]) -> Dict[str, int]:
    """
    Counts the total number of images and label files in the dataset.
    """
    stats = {'images': 0, 'labels': 0}
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    
    for subset in subsets:
        # Check for 'valid' or 'val'
        if subset == 'val' and not (root_dir / subset).exists():
            if (root_dir / 'valid').exists():
                subset = 'valid'
        
        label_dir = root_dir / subset / 'labels'
        img_dir = root_dir / subset / 'images'
        
        if label_dir.exists():
            stats['labels'] += len(list(label_dir.glob('*.txt')))
        
        if img_dir.exists():
            # Count files with valid image extensions
            for f in img_dir.iterdir():
                if f.suffix.lower() in img_exts:
                    stats['images'] += 1
                    
    return stats

def filter_yolo_dataset(root_path_str: str, classes_to_keep: List[int]):
    """
    Filters a YOLO dataset, keeping only specific classes, re-indexing them,
    and removing empty samples. Reports file counts before and after.
    """
    
    # 1. Setup Paths
    root_dir = Path(root_path_str).resolve()
    if not root_dir.exists():
        print(f"[Error] Root directory not found: {root_dir}")
        return

    print(f"{'='*40}")
    print(f"[*] Dataset Root: {root_dir}")
    
    # 2. Setup Class Mapping
    classes_to_keep = sorted(list(set(classes_to_keep)))
    idx_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(classes_to_keep)}
    
    print(f"[*] Keeping Classes (Old Indices): {classes_to_keep}")
    print(f"[*] Re-indexing Map: {idx_mapping}")

    subsets = ['train', 'val', 'test']
    
    # --- STEP 1: COUNT BEFORE ---
    print(f"\n[*] Counting files BEFORE filtering...")
    initial_stats = count_files(root_dir, subsets)
    print(f"    Images: {initial_stats['images']}")
    print(f"    Labels: {initial_stats['labels']}")

    # 3. Processing Loop
    stats = {'scanned': 0, 'deleted': 0, 'modified': 0}
    valid_img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

    for subset in subsets:
        # Handle 'val' vs 'valid' naming convention
        current_subset = subset
        if subset == 'val' and not (root_dir / subset).exists():
            if (root_dir / 'valid').exists():
                current_subset = 'valid'
        
        label_dir = root_dir / current_subset / 'labels'
        img_dir = root_dir / current_subset / 'images'
        
        if not label_dir.exists():
            continue

        print(f"\n--- Processing Subset: {current_subset} ---")
        
        label_files = list(label_dir.glob('*.txt'))
        
        for label_file in tqdm(label_files, desc=f"Filtering {current_subset}"):
            stats['scanned'] += 1
            
            with open(label_file, 'r') as f:
                lines = f.readlines()

            new_lines = []
            has_valid_class = False

            for line in lines:
                parts = line.strip().split()
                if not parts: continue
                
                try:
                    cls_id = int(parts[0])
                    # Check if class is in our whitelist
                    if cls_id in classes_to_keep:
                        # Remap to new index (0, 1, 2...)
                        new_id = idx_mapping[cls_id]
                        parts[0] = str(new_id)
                        new_lines.append(" ".join(parts) + "\n")
                        has_valid_class = True
                except ValueError:
                    continue

            # Decision: Delete or Update
            if not has_valid_class:
                # Delete Label
                label_file.unlink()
                
                # Delete Image
                stem = label_file.stem
                img_deleted = False
                for ext in valid_img_exts:
                    img_path = img_dir / (stem + ext)
                    if img_path.exists():
                        img_path.unlink()
                        break
                
                stats['deleted'] += 1
            else:
                # Rewrite file if content changed (optimization)
                original_content = "".join(lines)
                new_content = "".join(new_lines)
                
                if original_content != new_content:
                    with open(label_file, 'w') as f:
                        f.writelines(new_lines)
                    stats['modified'] += 1

    # --- STEP 2: COUNT AFTER ---
    print(f"\n[*] Counting files AFTER filtering...")
    final_stats = count_files(root_dir, subsets)

    # 4. Update data.yaml
    yaml_path = root_dir / 'data.yaml'
    new_names_list = []
    
    if yaml_path.exists():
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        old_names = data.get('names')
        
        # Robust Name Extraction
        for old_idx in classes_to_keep:
            name = None
            if isinstance(old_names, list):
                if old_idx < len(old_names):
                    name = old_names[old_idx]
            elif isinstance(old_names, dict):
                # Try int key first, then string key
                name = old_names.get(old_idx) or old_names.get(str(old_idx))
            
            # Fallback if name is missing in original YAML
            if name is None:
                name = f"class_{old_idx}"
            
            new_names_list.append(name)

        # Update YAML structure
        # Convert list to dict for safety (YOLO supports both, dict is explicit)
        data['names'] = {i: n for i, n in enumerate(new_names_list)}
        data['nc'] = len(new_names_list)
        
        # Fix paths to absolute to prevent future errors
        if 'train' in data: data['train'] = str(root_dir / 'train' / 'images')
        if 'val' in data: data['val'] = str(root_dir / 'valid' / 'images')
        if 'valid' in data: data['valid'] = str(root_dir / 'valid' / 'images')
        if 'test' in data: data['test'] = str(root_dir / 'test' / 'images')

        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, sort_keys=False)
            
        print(f"[*] Updated YAML: {yaml_path}")
        print(f"[*] New Class Names: {new_names_list}")

    # 5. Final Report
    print(f"\n{'='*40}")
    print(f"       SUMMARY REPORT       ")
    print(f"{'='*40}")
    print(f"Files Scanned:        {stats['scanned']}")
    print(f"Modified Labels:      {stats['modified']}")
    print(f"Deleted Samples:      {stats['deleted']} (Image + Label pairs)")
    print(f"{'-'*40}")
    print(f"Total Images Before:  {initial_stats['images']}")
    print(f"Total Images After:   {final_stats['images']}")
    print(f"Diff:                 {final_stats['images'] - initial_stats['images']}")
    print(f"{'-'*40}")
    print(f"Total Labels Before:  {initial_stats['labels']}")
    print(f"Total Labels After:   {final_stats['labels']}")
    print(f"Diff:                 {final_stats['labels'] - initial_stats['labels']}")
    print(f"{'='*40}")

def visualize_single_class_samples(image_dir, label_dir, yaml_path, samples_per_class=3):
    """
    Visualizes samples for every class, drawing ONLY the bounding box relevant 
    to that specific class (ignoring other objects in the same image).
    """
    
    # 1. Load Class Names from YAML
    if not os.path.exists(yaml_path):
        print(f"Error: YAML file not found at {yaml_path}")
        return

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Handle 'names' (list or dict)
    names_data = data.get('names', {})
    class_names = {}
    if isinstance(names_data, list):
        class_names = {i: name for i, name in enumerate(names_data)}
    elif isinstance(names_data, dict):
        class_names = {int(k): v for k, v in names_data.items()}

    # 2. Map Classes to Images
    class_to_images = defaultdict(list)
    valid_img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    print("[*] Scanning label files...")
    label_files = list(Path(label_dir).glob('*.txt'))
    
    # Pre-scan to group images by class
    for label_file in label_files:
        # Find corresponding image
        img_path = None
        for ext in valid_img_exts:
            potential_path = Path(image_dir) / (label_file.stem + ext)
            if potential_path.exists():
                img_path = str(potential_path)
                break
        
        if not img_path: continue

        # Read file to find which classes are present
        with open(label_file, 'r') as f:
            unique_classes_in_this_file = set()
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                try:
                    cls_id = int(parts[0])
                    if cls_id not in unique_classes_in_this_file:
                        class_to_images[cls_id].append(img_path)
                        unique_classes_in_this_file.add(cls_id)
                except ValueError:
                    continue

    # 3. Setup Plot
    available_classes = sorted(class_to_images.keys())
    num_classes = len(available_classes)
    
    if num_classes == 0:
        print("[!] No classes found.")
        return

    print(f"[*] Visualizing {num_classes} classes (ONLY target labels shown)...")

    # Dynamic figure height
    fig_height = num_classes * 3.5
    fig_width = samples_per_class * 4
    
    fig, axes = plt.subplots(nrows=num_classes, ncols=samples_per_class, 
                             figsize=(fig_width, fig_height), 
                             squeeze=False)

    # 4. Draw
    for row_idx, target_cls_id in enumerate(available_classes):
        # Get class name
        target_cls_name = class_names.get(target_cls_id, f"Class {target_cls_id}")
        
        # Get random samples for THIS class
        all_images = class_to_images[target_cls_id]
        n_samples = min(samples_per_class, len(all_images))
        selected_imgs = random.sample(all_images, n_samples)
        
        for col_idx in range(samples_per_class):
            ax = axes[row_idx, col_idx]
            ax.axis('off') # Hide axes

            if col_idx < n_samples:
                img_path = selected_imgs[col_idx]
                
                # Load Image
                img = cv2.imread(img_path)
                if img is None: continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, _ = img.shape
                
                # Draw Boxes
                label_path = os.path.join(label_dir, Path(img_path).stem + '.txt')
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        for line in f:
                            data = line.strip().split()
                            if len(data) >= 5:
                                box_cls_id = int(data[0])
                                
                                # --- KEY CHANGE IS HERE ---
                                # Only draw if the box class matches the current row's class
                                if box_cls_id == target_cls_id:
                                    cx, cy, bw, bh = map(float, data[1:5])
                                    
                                    x1 = int((cx - bw/2) * w)
                                    y1 = int((cy - bh/2) * h)
                                    x2 = int((cx + bw/2) * w)
                                    y2 = int((cy + bh / 2) * h)
                                    
                                    # Draw Box (Green)
                                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    
                                    # Draw Text
                                    label_text = class_names.get(box_cls_id, str(box_cls_id))
                                    cv2.putText(img, label_text, (x1, y1 - 8), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                ax.imshow(img)
                
                # Set Title only on the first column
                if col_idx == 0:
                    ax.set_title(f"ID: {target_cls_id} | {target_cls_name}", 
                                 fontsize=12, loc='left', color='blue', fontweight='bold')

    plt.tight_layout()
    plt.show()

def convert_seg_to_detection_in_place(dataset_root: str) -> None:
    """
    Converts YOLO instance segmentation labels to object detection bounding box format
    **in place** — directly overwrites the original label files.

    Only processes the standard YOLO dataset subfolders:
        - train/labels
        - valid/labels (or val/labels)
        - test/labels

    No other folders are touched.

    For each label file:
    - Reads segmentation polygons (class + normalized x1 y1 x2 y2 ... xn yn)
    - Computes the tightest axis-aligned bounding box (min/max coordinates)
    - Rewrites the file in standard YOLO detection format:
      class x_center y_center width height (all normalized)

    Args:
        dataset_root (str): Path to the root of the YOLO dataset folder
                            (should contain train/, valid/ or val/, test/, and data.yaml)

    Returns:
        None — modifies files directly on disk and prints summary

    Example:
        convert_seg_to_detection_in_place("/kaggle/working/my_dataset")
        convert_seg_to_detection_in_place("/path/to/your/dataset")
    """
    root = Path(dataset_root).resolve()
    if not root.exists() or not root.is_dir():
        print(f"Error: Dataset root directory not found → {root}")
        return

    # Only these standard subfolders will be processed
    possible_subsets = ["train", "valid", "val", "test"]

    print(f"Starting label conversion in: {root}")
    print("Only processing subfolders: train / valid / val / test\n")

    total_files_processed = 0
    files_converted = 0
    files_skipped = 0
    files_with_errors = 0

    for subset in possible_subsets:
        labels_dir = root / subset / "labels"

        if not labels_dir.exists() or not labels_dir.is_dir():
            print(f"→ Labels folder not found: {labels_dir}")
            continue

        print(f"\nProcessing: {subset}/labels")

        label_files = list(labels_dir.glob("*.txt"))
        if not label_files:
            print("   No .txt files found")
            continue

        for label_path in tqdm(label_files, desc=f"  {subset}"):
            total_files_processed += 1

            try:
                with open(label_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                new_lines = []
                file_was_modified = False

                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 5:  # class + at least 2 points (4 coords)
                        new_lines.append(line)
                        continue

                    class_id = parts[0]
                    coords = [float(x) for x in parts[1:]]

                    # Segmentation coords must be even (x,y pairs)
                    if len(coords) % 2 != 0:
                        new_lines.append(line)
                        continue

                    xs = coords[0::2]
                    ys = coords[1::2]

                    if not xs or not ys:
                        continue

                    # Calculate tight bounding box
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)

                    x_center = (x_min + x_max) / 2
                    y_center = (y_min + y_max) / 2
                    width = x_max - x_min
                    height = y_max - y_min

                    # Clamp values to valid [0, 1] range
                    x_center = max(0.0, min(1.0, x_center))
                    y_center = max(0.0, min(1.0, y_center))
                    width = max(0.0, min(1.0, width))
                    height = max(0.0, min(1.0, height))

                    # New detection-format line
                    new_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                    new_lines.append(new_line)
                    file_was_modified = True

                # Overwrite file only if content changed
                if file_was_modified:
                    with open(label_path, "w", encoding="utf-8") as f:
                        f.writelines(new_lines)
                    files_converted += 1
                else:
                    files_skipped += 1

            except Exception as e:
                print(f"  Error processing {label_path.name}: {e}")
                files_with_errors += 1

    print("\n" + "=" * 60)
    print("CONVERSION SUMMARY")
    print("=" * 60)
    print(f"Total label files processed : {total_files_processed}")
    print(f"Files converted/overwritten  : {files_converted}")
    print(f"Files skipped (no change)    : {files_skipped}")
    print(f"Files with errors            : {files_with_errors}")
    print("=" * 60)
