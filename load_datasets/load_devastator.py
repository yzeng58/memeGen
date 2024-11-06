import pandas as pd
import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from configs import get_dataset_dir
import pdb, re
from helper import read_json

def get_description_path(image_path: str, description: str):
    description_path = image_path.replace('/images/', f'/description/{description}/')
    description_path = re.sub(r'\.(jpeg|jpg|png|gif|bmp|webp)$', '.json', description_path, flags=re.IGNORECASE)
    return description_path

def load_devastator(description: str = '', eval_mode: str = ''):
    dataset = pd.read_csv(f'{get_dataset_dir("devastator")}/memes.csv')
    dataset = dataset[dataset["score"] >= 50]
    dataset = dataset[dataset['url'].str.split('/').str[-1].str.contains('\.', regex=True)]
    dataset['image_path'] = dataset[['id', 'url']].apply(lambda x: f'{get_dataset_dir("devastator")}/images/{x[0]}.{x[1].split("/")[-1].split(".")[-1]}', axis = 1)
    dataset["context"] = dataset["title"] + "\n" + dataset["body"].fillna("")
    dataset = dataset[dataset['image_path'].apply(os.path.exists)]
    dataset = dataset[['image_path', "context"]].reset_index(drop=True)
    if eval_mode == '': return dataset

    df = []
    for idx, row in dataset.iterrows():
        image_path = row['image_path']
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        candidate_path = os.path.join(f"{get_dataset_dir('devastator')}/candidates", f'{image_name}.json')
        if os.path.exists(candidate_path):
            candidate = read_json(candidate_path)
            if candidate['is_good'] and os.path.exists(candidate['candidate']) and os.path.exists(candidate['random']) and os.path.exists(image_path):
                df.append({
                    'ground_truth_path': image_path,
                    'context': row['context'],
                    'closest_candidate_path': candidate['candidate'],
                    'random_candidate_path': candidate['random']
                })

    df = pd.DataFrame(df)
    if eval_mode == 'single':
        expanded_df = []
        for _, row in df.iterrows():
            # Original image
            expanded_df.append({
                'image_path': row['ground_truth_path'],
                'context': row['context'],
                'label': 1
            })
            
            # Closest candidate
            expanded_df.append({
                'image_path': row['closest_candidate_path'], 
                'context': row['context'],
                'label': 0
            })
            
            # Random candidate
            expanded_df.append({
                'image_path': row['random_candidate_path'],
                'context': row['context'], 
                'label': 0
            })
            
        df = pd.DataFrame(expanded_df)

        if description:
            df["description_path"] = df['image_path'].apply(lambda x: get_description_path(x, description))
            df = df[['image_path', "context", "description_path"]]

    elif eval_mode == 'threeway':
        if description:
            df["ground_truth_description_path"] = df['ground_truth_path'].apply(lambda x: get_description_path(x, description))
            df["closest_candidate_description_path"] = df['closest_candidate_path'].apply(lambda x: get_description_path(x, description))
            df["random_candidate_description_path"] = df['random_candidate_path'].apply(lambda x: get_description_path(x, description))
            df = df[[
                "context", 
                'ground_truth_path', 
                "ground_truth_description_path", 
                'closest_candidate_path', 
                "closest_candidate_description_path", 
                'random_candidate_path', 
                "random_candidate_description_path"
            ]]
    else:
        raise ValueError(f"Invalid eval_mode: {eval_mode}")
    return df
