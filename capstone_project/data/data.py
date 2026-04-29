import os
import requests
import zipfile
import pandas as pd
from io import BytesIO


# download files and before it check if its present or not.
# it save the files using .txt or .data
# and it save with specific name
def download_and_extract(url, extract_to='data', expected_file=None):

    #
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    

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
    

url_a = "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip"


path_a = download_and_extract(url_a, expected_file="wdbc.data")


if path_a:
    df_a = pd.read_csv(path_a)
    print(df_a.head())
    print(df_a.count())


url_b = "https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip"

path_b = download_and_extract(url_b, expected_file="data_banknote_authentication.txt")

if path_b:
    df_b = pd.read_csv(path_b)
    print(df_b.head())
    print(df_b.count())