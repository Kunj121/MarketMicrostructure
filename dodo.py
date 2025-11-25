

import zipfile
from pathlib import Path

def unzip_files():
    """
    Unzip all .zip files inside the data/ folder and extract them into
    folders with matching names.

    Example:
        data/assignment4_datafiles.zip → data/assignment4_datafiles/
    """

    DATA_DIR = Path(__file__).resolve().parent / "data"
    zip_files = list(DATA_DIR.glob("*.zip"))

    if not zip_files:
        print("No zip files found in data/.")
        return

    for z in zip_files:
        extract_dir = DATA_DIR / z.stem
        extract_dir.mkdir(exist_ok=True)

        print(f"Extracting {z.name} → {extract_dir}/ ...")
        with zipfile.ZipFile(z, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

    print("✓ All zip files extracted successfully.")

if __name__ == "__main__":
    unzip_files()