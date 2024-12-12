import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from configs import get_dataset_dir
from helper import read_json
import pandas as pd

def load_british_complaints(train_test_split: bool = False):
    dataset_dir = get_dataset_dir('british_complaints')
    complaints = read_json(os.path.join(dataset_dir, 'complaints.json'))
    df = pd.Series(complaints)
    if train_test_split:
        train_complaints = df.sample(frac=0.5, random_state=42)
        test_complaints = df.drop(train_complaints.index)
        return {
            "train": train_complaints,
            "test": test_complaints,
        }
    else:
        return df