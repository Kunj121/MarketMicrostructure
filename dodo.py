

import zipfile
from pathlib import Path

def task_unzip_files():
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


def install_requirements():
    """
    Install required packages from requirements.txt.
    """
    import subprocess
    import sys

    requirements_file = Path(__file__).resolve().parent / "requirements.txt"
    if requirements_file.exists():
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])
    else:
        print("No requirements.txt file found.")

def _update_gan_learning_rates():
    """
    Ask the user for new learning rates and update settings.py
    for G_PARAMS['lr'] and D_PARAMS['lr'].

    This version is robust to spacing/formatting differences in settings.py:
    it replaces any line that starts with 'G_PARAMS' or 'D_PARAMS'.
    """
    settings_path = Path(__file__).parent / "settings.py"
    text = settings_path.read_text()

    print("=== Update GAN learning rates in settings.py ===")
    g_lr = input("Enter generator learning rate (G_PARAMS['lr']): ").strip()
    d_lr = input("Enter discriminator learning rate (D_PARAMS['lr']): ").strip()

    # Basic validation
    try:
        float(g_lr)
        float(d_lr)
    except ValueError:
        raise ValueError("Both learning rates must be numeric (float-compatible).")

    lines = text.splitlines()
    new_lines = []
    found_g = False
    found_d = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("G_PARAMS"):
            new_lines.append(f"G_PARAMS = {{'lr': {g_lr}, 'betas': (0.99, 0.999)}}")
            found_g = True
        elif stripped.startswith("D_PARAMS"):
            new_lines.append(f"D_PARAMS = {{'lr': {d_lr},   'betas': (0.99, 0.999)}}")
            found_d = True
        else:
            new_lines.append(line)

    if not (found_g and found_d):
        raise RuntimeError(
            "Could not find lines starting with 'G_PARAMS' and 'D_PARAMS' in settings.py.\n"
            "Make sure those variables exist in the file."
        )

    settings_path.write_text("\n".join(new_lines))
    print(f"Updated learning rates: G_PARAMS['lr']={g_lr}, D_PARAMS['lr']={d_lr}")


# doit task: run GAN full pipeline
def task_run_gan_full():
    """
    doit task:
      1) Ask user for GAN learning rates and update settings.py
      2) Run LOB_GAN_training.py
      3) Run assignment_4.ipynb
    """
    return {
        "actions": [
            (lambda: print("→ Updating learning rates in settings.py ..."),),
            (_update_gan_learning_rates,),

            (lambda: print("→ Running LOB_GAN_training.py ..."),),
            "python src/LOB_GAN_training.py",

            # (lambda: print("→ Executing assignment_4.ipynb ..."),),
            # "jupyter nbconvert --to notebook --execute src/assignment_4.ipynb --inplace",

            (lambda: print("✓ task_run_gan_full completed."),),
        ],
        "verbosity": 2,
    }
