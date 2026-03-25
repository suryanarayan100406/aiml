"""
Download real datasets for ANI Flow Optimizer training.
Supports: RAVDESS, LibriSpeech (subset), COCO (filtered classes).

Usage:
  python download_real_datasets.py --ravdess
  python download_real_datasets.py --librispeech
  python download_real_datasets.py --coco
  python download_real_datasets.py --all
"""
import os, argparse, urllib.request, zipfile, tarfile, sys
from tqdm import tqdm

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "raw")

class DownloadProgress(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_file(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        print(f"  Already exists: {dest}")
        return
    print(f"  Downloading: {url}")
    with DownloadProgress(unit='B', unit_scale=True, miniters=1, desc=os.path.basename(dest)) as t:
        urllib.request.urlretrieve(url, filename=dest, reporthook=t.update_to)

def download_ravdess():
    """Download RAVDESS emotional speech dataset from Zenodo."""
    print("\n🎙️ Downloading RAVDESS dataset...")
    dest_dir = os.path.join(RAW_DIR, "ravdess_audio")
    os.makedirs(dest_dir, exist_ok=True)
    url = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
    zip_path = os.path.join(dest_dir, "ravdess.zip")
    download_file(url, zip_path)
    print("  Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(dest_dir)
    print(f"  ✅ RAVDESS extracted to {dest_dir}")

def download_librispeech():
    """Download LibriSpeech train-clean-100 subset."""
    print("\n📚 Downloading LibriSpeech train-clean-100...")
    dest_dir = os.path.join(RAW_DIR, "librispeech_subset")
    os.makedirs(dest_dir, exist_ok=True)
    url = "https://www.openslr.org/resources/12/train-clean-100.tar.gz"
    tar_path = os.path.join(dest_dir, "train-clean-100.tar.gz")
    download_file(url, tar_path)
    print("  Extracting (this may take a while)...")
    with tarfile.open(tar_path, 'r:gz') as t:
        t.extractall(dest_dir)
    print(f"  ✅ LibriSpeech extracted to {dest_dir}")

def download_coco_annotations():
    """Download COCO 2017 annotations (not images — they're too large)."""
    print("\n🖼️ Downloading COCO 2017 annotations...")
    dest_dir = os.path.join(RAW_DIR, "coco_filtered")
    os.makedirs(dest_dir, exist_ok=True)
    url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    zip_path = os.path.join(dest_dir, "annotations.zip")
    download_file(url, zip_path)
    print("  Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(dest_dir)
    print(f"  ✅ COCO annotations extracted to {dest_dir}")
    print("  Note: Download images separately with pycocotools or use synthetic data.")

def main():
    parser = argparse.ArgumentParser(description="Download real datasets")
    parser.add_argument("--ravdess", action="store_true")
    parser.add_argument("--librispeech", action="store_true")
    parser.add_argument("--coco", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if not any([args.ravdess, args.librispeech, args.coco, args.all]):
        parser.print_help()
        sys.exit(1)

    if args.all or args.ravdess:
        download_ravdess()
    if args.all or args.librispeech:
        download_librispeech()
    if args.all or args.coco:
        download_coco_annotations()

    print("\n✅ Done! Datasets saved to:", os.path.abspath(RAW_DIR))

if __name__ == "__main__":
    main()
