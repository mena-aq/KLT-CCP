import os
import requests
from tqdm import tqdm
import zipfile

# === CONFIGURATION ===
DATASET_FILES = {
    "images_laptops.zip": "https://huggingface.co/datasets/FatimaSohailll/PPM-Image-Dataset-for-KLT-Feature-Tracking/resolve/main/images_laptops.zip",
    "images_traffic.zip": "https://huggingface.co/datasets/FatimaSohailll/PPM-Image-Dataset-for-KLT-Feature-Tracking/resolve/main/images_traffic.zip"
}
DATA_DIR = "../../data"

def download_file(url, output_path):
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(output_path, 'wb') as file, tqdm(
        desc=f"Downloading {os.path.basename(output_path)}",
        total=total,
        unit='iB', unit_scale=True, unit_divisor=1024
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def extract_zip(zip_path, extract_to):
    """Extract zip file to directory."""
    print(f"\nExtracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to '{extract_to}'")

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    for filename, url in DATASET_FILES.items():
        zip_path = os.path.join(DATA_DIR, filename)

        # Download if not present
        if not os.path.exists(zip_path):
            print(f"\nDownloading {filename} from Hugging Face...")
            download_file(url, zip_path)
        else:
            print(f"{filename} already exists, skipping download.")

        # Extract the file
        folder_name = filename.replace(".zip", "")
        extract_path = os.path.join(DATA_DIR, folder_name)

        if not os.path.exists(extract_path):
            extract_zip(zip_path, extract_path)
        else:
            print(f"{folder_name} already extracted, skipping.")

    print("\nAll datasets are ready in the 'data/' folder!")

if __name__ == "__main__":
    main()

