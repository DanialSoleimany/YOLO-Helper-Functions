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
