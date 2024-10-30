import pandas as pd
import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from configs import get_dataset_dir
from helper import read_jsonl
import pdb, re

def get_img_path(path):
    return f'{get_dataset_dir("meta_hateful")}/images/{os.path.basename(path)}'

def get_description_path(image_path: str, description: str):
    description_path = image_path.replace('/images/', f'/description/{description}/')
    description_path = re.sub(r'\.(jpeg|jpg|png|gif|bmp|webp)$', '.json', description_path, flags=re.IGNORECASE)
    return description_path

def load_meta_hateful(description: str = ''):
    data = read_jsonl(f'{get_dataset_dir("meta_hateful")}/train.jsonl')
    data['image_path'] = data['img'].apply(get_img_path)
    data = data[['image_path', 'text', 'label']]

    if description:
        data['description_path'] = data['image_path'].apply(lambda x: get_description_path(x, description))
        data = data[['image_path', 'text', 'label', 'description_path']]
    return data
