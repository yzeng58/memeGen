import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from configs import get_dataset_dir
from helper import read_json

def load_ours_gen_v1():
    dataset_dir = get_dataset_dir('ours_gen_v1')
    complaints = read_json(os.path.join(dataset_dir, 'complaints.json'))
    
    return complaints