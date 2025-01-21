import pandas as pd
import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from configs import get_dataset_dir

def load_llm_meme(
    description: str = '', 
    train_test_split: bool = False,
    difficulty: str = 'easy',
    score_analysis: bool = False,
):
    if not difficulty in ["easy"]: 
        raise ValueError(f'Difficulty {difficulty} not supported for llm_meme dataset, please choose from [easy]')
    
    if not description in ["", "default"]:
        raise ValueError(f"For llm_meme dataset, please choose description from ['', 'default']")
    
    data_df = pd.read_csv(f"{get_dataset_dir('llm_meme')}/dataset.csv")
    data_df["image_path"] = data_df["image_path"].apply(lambda x: f"{get_dataset_dir('llm_meme')}/images/{x}")
    data_df["description_path"] = data_df["description_path"].apply(lambda x: f"{get_dataset_dir('llm_meme')}/images/{x}")
    df = data_df[[
        'image_path', 
        'label', 
        'description_path',
    ]]
    
    if train_test_split:
        train_df = df.sample(frac=0.5, random_state=42)
        test_df = df.drop(train_df.index)
        return {
            "train": train_df.reset_index(drop=True), 
            "test": test_df.reset_index(drop=True),
        }
    else:
        return df
