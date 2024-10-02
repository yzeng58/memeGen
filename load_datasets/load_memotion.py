import pandas as pd
import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from configs import dataset_dir


def load_memotion():
    memotion_dir = f'{dataset_dir}/memotion_dataset_7k'
    memotion_labels = pd.read_csv(f'{memotion_dir}/labels.csv')

    humor_mapping = {
        'not_funny': 1,
        'funny': 2,
        'very_funny': 3,
        'hilarious': 4
    }

    # Apply the mapping to create a new column 'humor_level'
    memotion_labels['humor_level'] = memotion_labels['humour'].map(humor_mapping)
    df = memotion_labels[['image_name', 'humor_level', 'text_corrected']]
    df.columns = ['image_name', 'humor_level', 'text']
    df = df.copy()  # Create a copy to avoid SettingWithCopyWarning
    df['image_path'] = df['image_name'].apply(lambda img_name: f'{memotion_dir}/images/{img_name}')
    df['humor_level'] = df['humor_level'].astype(int)
    return df[['image_path', 'text', 'humor_level']]
