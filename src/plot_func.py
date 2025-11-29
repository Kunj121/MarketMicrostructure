import os
import re
import math
from typing import List, Optional, Union

import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]   # the repo root
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from settings import DATA_DIR, OUTPUT_DIR, G_PARAMS,TRAINING_PLOT_DIR


import matplotlib.pyplot as plt
from PIL import Image


# ---------- Indexer: looks only in PARAMS_* folders -------------------------

def _index_loss_plots(base_dir: str):
    """
    Look in base_dir/PARAMS_* and return a list of dicts:
    {ticker, g_lr, d_lr, path}.
    """
    # Handles 0.001, 0.0002, 5e-05, etc.
    pattern = re.compile(
        r"(?P<ticker>\d{4})_loss_plot_G_lr(?P<g_lr>[0-9.e-]+)_D_lr(?P<d_lr>[0-9.e-]+)\.png$"
    )

    records = []

    # only scan subfolders named PARAMS_*
    for entry in os.scandir(base_dir):
        if not (entry.is_dir() and entry.name.upper().startswith("PARAMS_")):
            continue

        for fname in os.listdir(entry.path):
            if not fname.endswith(".png"):
                continue

            m = pattern.search(fname)
            if not m:
                continue

            records.append(
                {
                    "ticker": m.group("ticker"),
                    "g_lr": float(m.group("g_lr")),
                    "d_lr": float(m.group("d_lr")),
                    "path": os.path.join(entry.path, fname),
                }
            )

    return records


# ---------- Main plotting function ------------------------------------------

def plot_loss_curves(
    base_dir: str,
    tickers: Optional[Union[str, List[str]]] = None,
    g_lrs: Optional[Union[float, List[float]]] = None,
    d_lrs: Optional[Union[float, List[float]]] = None,
    max_cols: int = 3,
):
    """
    Show loss plots filtered by ticker and/or learning rates.

    base_dir should be the *training_plots* folder that contains PARAMS_1, PARAMS_2, ...

    Examples
    --------
    # Same LR across tickers:
    plot_loss_curves(
        base_dir="~/Downloads/MarketMicrostructure/plots/training_plots",
        g_lrs=0.001,
        d_lrs=0.001,
    )

    # Same ticker, different LRs:
    plot_loss_curves(
        base_dir="~/Downloads/MarketMicrostructure/plots/training_plots",
        tickers="0050",
    )
    """
    records = _index_loss_plots(os.path.expanduser(base_dir))

    if isinstance(tickers, str):
        tickers = [tickers]
    if isinstance(g_lrs, (float, int)):
        g_lrs = [float(g_lrs)]
    if isinstance(d_lrs, (float, int)):
        d_lrs = [float(d_lrs)]

    def _keep(rec):
        if tickers is not None and rec["ticker"] not in tickers:
            return False
        if g_lrs is not None and rec["g_lr"] not in g_lrs:
            return False
        if d_lrs is not None and rec["d_lr"] not in d_lrs:
            return False
        return True

    matches = [r for r in records if _keep(r)]

    if not matches:
        raise ValueError("No plots matched the given filters.")

    n = len(matches)
    cols = min(max_cols, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))

    # Normalize axes to 2D list
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    flat_axes = [ax for row in axes for ax in row]

    for ax in flat_axes:
        ax.axis("off")

    for ax, rec in zip(flat_axes, matches):
        img = Image.open(rec["path"])
        ax.imshow(img)
        ax.set_title(
            f"{rec['ticker']} | G_lr={rec['g_lr']} | D_lr={rec['d_lr']}",
            fontsize=9,
        )
        ax.axis("off")

    plt.tight_layout()
    plt.show()