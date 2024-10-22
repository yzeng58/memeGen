import pandas as pd
import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from configs import get_dataset_dir

def load_130k():
    dataset_dir = get_dataset_dir('130k')

    image_dir = f'{dataset_dir}/images'
    files = os.listdir(image_dir)
    image_paths = []
    for file in files:
        if file.endswith('.DS_Store'): continue
        image_paths.append(f'{image_dir}/{file}')

    df = pd.DataFrame({'image_path': image_paths})
    return df

