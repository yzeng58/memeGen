import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

import wandb
from environment import WANDB_INFO

api = wandb.Api()

runs = api.runs(f"{WANDB_INFO['entity']}/{WANDB_INFO['project']}")

for run in runs:
    if not 'eval_mode' in run.config:
        print(run.config)
        run.config['eval_mode'] = 'single'
        run.update()