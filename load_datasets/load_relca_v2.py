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

def load_relca_v2(
    description: str = '', 
    train_test_split: bool = False,
    difficulty: str = 'easy',
    score_analysis: bool = False,
):
    if not difficulty in ["easy"]: 
        raise ValueError(f'Difficulty {difficulty} not supported for version 2, please choose from [easy]')
        
    if score_analysis:
        data_df = pd.read_csv(f"{get_dataset_dir('relca')}/score_analysis_final_v2.csv")
        data_df["image_path"] = data_df["image_name"].apply(lambda x: f"{get_dataset_dir('relca')}/images/{x}")
        df = data_df[[
            'image_path', 
            'label', 
            'Q1_option', 
            'Q1_reasoning',
            'Q2_option', 
            'Q2_reasoning',
            'Q3_option', 
            'Q3_reasoning',
            'Q4_option', 
            'Q4_reasoning',
            'Q5_option', 
            'Q5_reasoning',
        ]]
    else: 
        data_df = pd.read_csv(f"{get_dataset_dir('relca')}/new_labels.csv")
        data_df["image_path"] = data_df["original_image"].apply(lambda x: f"{get_dataset_dir('relca')}/images/{x}")
        df = data_df[['image_path', 'label']]

    if description:
        df['description_path'] = df['image_path'].apply(lambda x: get_description_path(x, description))
        df = df[['image_path', 'label', 'description_path']]


    if train_test_split:
        train_df = df.sample(frac=0.5, random_state=42)
        test_df = df.drop(train_df.index)
        return {
            "train": train_df.reset_index(drop=True), 
            "test": test_df.reset_index(drop=True),
        }
    else:
        return df
    