import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from configs import get_dataset_dir

def load_ours_gen():
    dataset_dir = get_dataset_dir('ours_gen_v1')
    sys.path.append(dataset_dir)
    from complaints import complaints
    
    return complaints