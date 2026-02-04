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
