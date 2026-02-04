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
