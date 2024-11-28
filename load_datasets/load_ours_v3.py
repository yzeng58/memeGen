import pandas as pd
import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from configs import get_dataset_dir
import glob, re, pdb

def get_description_path(image_path: str, description: str):
    description_path = image_path.replace('/images/', '/description/').replace('/funny', f'/{description}').replace('/not_funny', f'/{description}')
    description_path = re.sub(r'\.(jpeg|jpg|png|gif|bmp|webp)$', '.json', description_path, flags=re.IGNORECASE)
    return description_path

def load_ours_v3(
    description: str = '', 
    train_test_split: bool = False,
    difficulty: str = 'easy',
):
    files = glob.glob(f'{get_dataset_dir("ours_v3")}/images/*/*')
    data = []
    for file in files:
        category = file.split('/')[-2]
        label = 1 if category == 'funny' else 0
        file_dict = {'image_path': file, 'label': label}

        if description: 
            description_path = get_description_path(file, description)
            file_dict['description_path'] = description_path
        data.append(file_dict)
    return pd.DataFrame(data)
