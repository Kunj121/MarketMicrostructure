import os
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / 'data' / "assignment4_datafiles"

file = DATA_DIR / "0056_md_202310_202310.csv.gz"

df = pd.read_csv(file)
