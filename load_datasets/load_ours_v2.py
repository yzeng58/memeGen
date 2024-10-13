import pandas as pd
import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from configs import dataset_dir
import glob

def load_ours_v2():
    files = glob.glob(f'{dataset_dir}/ours_v2/images/*/*')
    data = []
    for file in files:
        category = file.split('/')[-2]
        label = 1 if category == 'funny' else 0
        data.append({'image_path': file, 'label': label})

    return pd.DataFrame(data)