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
    
    col_names = ['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 
                 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean',
                 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se',
                 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se',
                 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst',
                 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 
                 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst']
    df_a = pd.read_csv(path_a, header=None, names=col_names)
    print(df_a.head())
    print(df_a.count())
    print(df_a.info())


url_b = "https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip"

path_b = download_and_extract(url_b, expected_file="data_banknote_authentication.txt")

if path_b:
    
    col_names_b = ['variance', 'skewness', 'curtosis', 'entropy', 'class']
    df_b = pd.read_csv(path_b, header=None, names=col_names_b)
    print(df_b.head())
    print("==================")
    print(df_b.count())
    print("==================")
    print(df_b.info())