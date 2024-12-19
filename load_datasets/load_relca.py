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

def load_relca(
    description: str = '', 
    train_test_split: bool = False,
    difficulty: str = 'easy',
    version: str = 'v1',
):
    if version == 'v2':
        if not difficulty in ["easy"]: 
            raise ValueError(f'Difficulty {difficulty} not supported for version {version}, please choose from [easy]')
        data_df = pd.read_csv(f"{get_dataset_dir('relca')}/new_labels.csv")
        data_df["image_path"] = data_df["original_image"].apply(lambda x: f"{get_dataset_dir('relca')}/images/{x}")
        df = data_df[['image_path', 'label']]
    else:
        data_df = pd.read_csv(f"{get_dataset_dir('relca')}/seen_dataset_postprocessed.csv")

        if difficulty == 'easy':
            score_threshold = .5
            upvote_threshold = (100, 10)
        elif difficulty == 'medium':
            score_threshold = .3
            upvote_threshold = (70, 30)
        elif difficulty == 'hard':
            score_threshold = .1
            upvote_threshold = (70, 30)
        else:
            raise ValueError(f'Difficulty {difficulty} not supported, please choose from [easy, medium, hard]')
            
        df = []
        for i, row in data_df.iterrows():
            if row["bws_score"] > score_threshold and int(row['upvote'].replace(',', '')) > upvote_threshold[0]:
                image_path = f'{get_dataset_dir("relca")}/images/{i}_{row["filename"]}'
                file_dict = {'image_path': image_path, 'label': 1}
                df.append(file_dict)
            elif row["bws_score"] < -score_threshold and int(row['upvote'].replace(',', '')) < upvote_threshold[1]:
                image_path = f'{get_dataset_dir("relca")}/images/{i}_{row["filename"]}'
                file_dict = {'image_path': image_path, 'label': 0}
                df.append(file_dict)

        df = pd.DataFrame(df)

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
    