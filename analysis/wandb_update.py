import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

import wandb
from environment import WANDB_INFO

api = wandb.Api()

runs = api.runs(f"{WANDB_INFO['entity']}/{WANDB_INFO['project']}")

for run in runs:
    # if not 'description' in run.config:
    #     print(run.config)
    #     run.config['description'] = ''
    #     run.update()
    if run.config['prompt_name'] == 'fs':
        print(run.config)
        run.config['prompt_name'] = 'standard'
        run.update()

    if run.config['prompt_name'] == 'cot_default':
        print(run.config)
        run.config['prompt_name'] = 'cot'
        run.update()
