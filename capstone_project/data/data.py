


import os
import zipfile
from io import BytesIO

import pandas as pd
import requests


SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))  
PROJECT_DIR  = os.path.dirname(SCRIPT_DIR)                    
RIPT_DIR     = os.path.join(PROJECT_DIR, "ript")          
TEMP_DIR     = os.path.join(SCRIPT_DIR,  "_tmp")    

os.makedirs(RIPT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)



def download_zip(url: str, extract_to: str, raw_filename: str) -> str | None:
    
    dest = os.path.join(extract_to, raw_filename)

    if os.path.exists(dest):
        print(f"  [skip] {raw_filename} already extracted.")
        return dest

    print(f"  Downloading {url} ...")
    response = requests.get(url, timeout=30)

    if response.status_code != 200:
        print(f"  [error] HTTP {response.status_code}")
        return None

    with zipfile.ZipFile(BytesIO(response.content)) as z:
        z.extractall(extract_to)
    print(f"  Extracted → {dest}")
    return dest



WDBC_CSV = os.path.join(RIPT_DIR, "wdbc.csv")

if os.path.exists(WDBC_CSV):
    print(f"[skip] wdbc.csv already exists at {WDBC_CSV}")
    print(pd.read_csv(WDBC_CSV).head())
else:
    raw = download_zip(
        url="https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip",
        extract_to=TEMP_DIR,
        raw_filename="wdbc.data",
    )

    if raw:
        col_names = [
            "id", "diagnosis",
            "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
            "smoothness_mean", "compactness_mean", "concavity_mean",
            "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
            "radius_se", "texture_se", "perimeter_se", "area_se",
            "smoothness_se", "compactness_se", "concavity_se",
            "concave_points_se", "symmetry_se", "fractal_dimension_se",
            "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
            "smoothness_worst", "compactness_worst", "concavity_worst",
            "concave_points_worst", "symmetry_worst", "fractal_dimension_worst",
        ]

        df_a = pd.read_csv(raw, header=None, names=col_names)

        # Drop id column; encode diagnosis M→1, B→0
        df_a = df_a.drop(columns=["id"])
        df_a["diagnosis"] = df_a["diagnosis"].map({"M": 1, "B": 0})

        df_a.to_csv(WDBC_CSV, index=False)
        print(f"[ok] wdbc.csv  — {len(df_a)} rows → {WDBC_CSV}")
        print(df_a.head())


# ── Dataset B — Banknote Authentication ───────────────────────────────────
BANKNOTE_CSV = os.path.join(RIPT_DIR, "banknote.csv")

if os.path.exists(BANKNOTE_CSV):
    print(f"[skip] banknote.csv already exists at {BANKNOTE_CSV}")
    print(pd.read_csv(BANKNOTE_CSV).head())
else:
    raw = download_zip(
        url="https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip",
        extract_to=TEMP_DIR,
        raw_filename="data_banknote_authentication.txt",
    )

    if raw:
        col_names_b = ["variance", "skewness", "curtosis", "entropy", "class"]

        df_b = pd.read_csv(raw, header=None, names=col_names_b)

        df_b.to_csv(BANKNOTE_CSV, index=False)
        print(f"[ok] banknote.csv — {len(df_b)} rows → {BANKNOTE_CSV}")
        print(df_b.head())


# ── Summary ────────────────────────────────────────────────────────────────
print(f"\nDatasets directory: {RIPT_DIR}")
for fname in ("wdbc.csv", "banknote.csv"):
    fpath = os.path.join(RIPT_DIR, fname)
    if os.path.exists(fpath):
        df = pd.read_csv(fpath)
        print(f"  {fname}: {df.shape[0]} rows × {df.shape[1]} cols")
        