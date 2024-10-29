import pandas as pd
import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from configs import get_dataset_dir
import re

def get_description_path(image_path: str, description: str):
    description_path = image_path.replace('/images/', f'/description/{description}/')
    description_path = re.sub(r'\.(jpeg|jpg|png|gif|bmp|webp)$', '.json', description_path, flags=re.IGNORECASE)
    return description_path

def load_relca(description: str = ''):
    data_df = pd.read_csv(f"{get_dataset_dir('relca')}/seen_dataset_postprocessed.csv")

    df = []
    for i, row in data_df.iterrows():
        if abs(row["bws_score"]) > 0.5:
            image_path = f'{get_dataset_dir("relca")}/images/{i}_{row["filename"]}'
            file_dict = {'image_path': image_path, 'label': row['bws_score'] > 0}
            df.append(file_dict)

    df = pd.DataFrame(df)
    if description:
        df['description_path'] = df['image_path'].apply(lambda x: get_description_path(x, description))
        df = df[['image_path', 'label', 'description_path']]
    return df
