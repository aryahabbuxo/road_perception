"""Creates clean zip archives for each processed dataset for Google Drive upload."""
import os
import shutil
import zipfile

DATA_DIR = "data_processed"
OUT_DIR = os.path.join(DATA_DIR, "zips")
os.makedirs(OUT_DIR, exist_ok=True)

for name in sorted(os.listdir(DATA_DIR)):
    folder = os.path.join(DATA_DIR, name)
    if not os.path.isdir(folder) or name == "zips":
        continue

    zip_path = os.path.join(OUT_DIR, f"{name}.zip")
    print(f"Zipping {name} -> {zip_path}...")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(folder):
            for file in files:
                file_path = os.path.join(root, file)
                # Archive relative to dataset folder (e.g. images/..., splits.json at root)
                arcname = os.path.relpath(file_path, folder)
                zf.write(file_path, arcname)

    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    shutil.copy2(zip_path, f"{name}.zip")
    print(f"  Done: {name}.zip ({size_mb:.1f} MB)")

print("\nAll dataset zip archives created in data_processed/zips/ ready for Drive upload!")
