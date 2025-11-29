import os
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / 'data' / "assignment4_datafiles"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'output_train'
TRAINING_PLOT_DIR = Path(__file__).resolve().parent / 'plots'/ 'training_plots'

# file = DATA_DIR / "0056_md_202310_202310.csv.gz"

# params (for display + filename)
G_PARAMS = {'lr': 0.00375, 'betas': (0.99, 0.999)}
D_PARAMS = {'lr': 0.00115,   'betas': (0.99, 0.999)}