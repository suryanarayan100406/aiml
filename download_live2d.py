import urllib.request
import os
import sys

def download_file(url, dest):
    print(f"Downloading {os.path.basename(dest)}...")
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'}
        )
        with urllib.request.urlopen(req) as response, open(dest, 'wb') as out_file:
            out_file.write(response.read())
        size = os.path.getsize(dest)
        print(f"  [OK] Saved {size // 1024} KB")
    except Exception as e:
        print(f"  [ERROR] {e}")

def main():
    vendor_dir = os.path.join("frontend", "vendor")
    os.makedirs(vendor_dir, exist_ok=True)

    files = [
        (
            "https://cdnjs.cloudflare.com/ajax/libs/pixi.js/7.4.0/pixi.min.js",
            os.path.join(vendor_dir, "pixi.min.js")
        ),
        (
            "https://cubism.live2d.com/sdk-web/cubismcore/live2dcubismcore.min.js",
            os.path.join(vendor_dir, "live2dcubismcore.min.js")
        ),
        (
            "https://cdn.jsdelivr.net/npm/pixi-live2d-display/dist/cubism4.min.js",
            os.path.join(vendor_dir, "cubism4.min.js")
        )
    ]

    print("Fetching Live2D Vendor Libraries...")
    for url, dest in files:
        download_file(url, dest)
        
    print("\nAll done! You can now launch / refresh the dashboard.")

if __name__ == "__main__":
    main()
