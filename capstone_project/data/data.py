import os
import requests
import zipfile
import pandas as pd
from io import BytesIO

def download_and_extract(url, extract_to='data', expected_file=None):
    """
    Downloads a ZIP from a URL and extracts its contents.
    Skips download if the expected_file already exists.
    """
    # Create the directory if it doesn't exist
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    
    # Check if we already have the specific data file
    if expected_file and os.path.exists(os.path.join(extract_to, expected_file)):
        print(f"File '{expected_file}' already exists. Skipping download.")
        return os.path.join(extract_to, expected_file)

    print(f"Downloading from {url}...")
    response = requests.get(url)
    
    if response.status_code == 200:
        with zipfile.ZipFile(BytesIO(response.content)) as z:
            z.extractall(extract_to)
            print("Extraction complete.")
            return os.path.join(extract_to, expected_file) if expected_file else extract_to
    else:
        print(f"Failed to download. Status code: {response.status_code}")
        return None