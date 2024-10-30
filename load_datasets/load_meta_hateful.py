import pandas as pd
import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from configs import get_dataset_dir
from helper import read_json
import pdb


def load_meta_hateful(description: str = ''):
    data = read_json(f'{get_dataset_dir("meta_hateful")}/train.jsonl')
    data = pd.DataFrame(data)
    return data