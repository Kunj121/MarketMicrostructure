# LOB GAN Training

This repository trains a GAN on limit order book (LOB) data for selected Taiwan stocks.

## 1. Setup & Installation

First, make sure you have Python 3.10+ installed.

(Optional but recommended) Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate        # Windows
```

Install req

```bash
pip install -r requirements.txt
```

2. Unzip the raw data files

The raw LOB data lives under data/assignment4_datafiles in compressed form.

Use the dodo.py task to unzip all files into the proper directory:


```bash

doit unzip_files
```
3. Run full GAN training

To run the full GAN training pipeline (for all stocks) via dodo.py, use the run_gan_full task.
```bash

doit run_gan_full
```
input your learning rate params

