import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from configs import get_dataset_dir
from helper import read_json
import pandas as pd

def load_isarcasm(train_test_split: bool = False):
    dataset_dir = get_dataset_dir('isarcasm')
    tweets = read_json(os.path.join(dataset_dir, 'tweets.json'))
    df = pd.Series(tweets)

    if train_test_split:
        train_tweets = df.sample(frac=0.5, random_state=42)
        test_tweets = df.drop(train_tweets.index)
        return {
            "train": train_tweets,
            "test": test_tweets,
        }
    else:
        return df