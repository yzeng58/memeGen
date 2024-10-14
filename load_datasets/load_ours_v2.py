import pandas as pd
import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from configs import dataset_dir
import glob, re

def load_ours_v2(description: str = ''):
    files = glob.glob(f'{dataset_dir}/ours_v2/images/*/*')
    data = []
    for file in files:
        category = file.split('/')[-2]
        label = 1 if category == 'funny' else 0
        file_dict = {'image_path': file, 'label': label}

        if description: 
            description_path = file.replace('/images/', '/description/').replace('/funny', f'/{description}').replace('/not_funny', f'/{description}')
            description_path = re.sub(r'\.(jpeg|jpg|png|gif|bmp|webp)$', '.json', description_path)
            file_dict['description_path'] = description_path

        data.append(file_dict)
    return pd.DataFrame(data)