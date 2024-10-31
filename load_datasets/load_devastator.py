import pandas as pd
import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from configs import get_dataset_dir
import pdb, re

def get_description_path(image_path: str, description: str):
    description_path = image_path.replace('/images/', f'/description/{description}/')
    description_path = re.sub(r'\.(jpeg|jpg|png|gif|bmp|webp)$', '.json', description_path, flags=re.IGNORECASE)
    return description_path

def load_devastator(description: str = ''):
    dataset = pd.read_csv(f'{get_dataset_dir("devastator")}/memes.csv')
    dataset = dataset[dataset["score"] >= 50]
    dataset = dataset[dataset['url'].str.split('/').str[-1].str.contains('\.', regex=True)]
    dataset['image_path'] = dataset[['id', 'url']].apply(lambda x: f'{get_dataset_dir("devastator")}/images/{x[0]}.{x[1].split("/")[-1].split(".")[-1]}', axis = 1)
    dataset["context"] = dataset["title"] + "\n" + dataset["body"].fillna("")
    dataset = dataset[dataset['image_path'].apply(os.path.exists)]
    df = dataset[['image_path', "context"]].reset_index(drop=True)

    if description:
        df["description_path"] = df['image_path'].apply(lambda x: get_description_path(x, description))
        df = df[['image_path', "context", "description_path"]]
    return df
