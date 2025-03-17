import pandas as pd
import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from configs import get_dataset_dir
import re, pdb

def get_description_path(image_path: str, description: str):
    description_path = image_path.replace('/images/', f'/description/{description}/')
    description_path = re.sub(r'\.(jpeg|jpg|png|gif|bmp|webp)$', '.json', description_path, flags=re.IGNORECASE)
    return description_path

def load_basic(
    description: str = '',
    train_test_split: bool = False,
):
    df = pd.read_csv(f'{get_dataset_dir("basic")}/meme_dataset.csv')
    df['image_path'] = df['image_path'].apply(lambda x: os.path.join(get_dataset_dir("basic"), x))

    if description:
        df['description_path'] = df['image_path'].apply(lambda x: get_description_path(x, description))
    else:
        df = df.drop('description_path', axis=1, errors='ignore')

    if train_test_split:
        train_df = df.sample(frac=0.5, random_state=42)
        test_df = df.drop(train_df.index)
        return {
            "train": train_df.reset_index(drop=True), 
            "test": test_df.reset_index(drop=True),
        }
    else:
        return df
