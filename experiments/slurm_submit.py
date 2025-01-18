import pdb
import pandas as pd
import os, sys
from copy import deepcopy
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from configs import get_peft_variant_name

configs = pd.read_csv(f"{root_dir}/experiments/configs.csv")

def create_python_eval_command(row):
    python_command = "python evaluation.py"
    for col in row.keys():
        if col in ["gpu_request", "experiment"]:
            continue
        elif pd.isna(row[col]):
            python_command += f" --{col} ''"
        elif col in ["wandb", "overwrite", "not_load_model"]:
            if row[col]: python_command += f" --{col}"
        else:
            python_command += f" --{col} {row[col]}"
    return python_command

def create_python_finetune_command(row):
    python_command = "python finetune.py"
    for col in row.keys():
        if col in ["gpu_request", "experiment", "wandb", "n_pairs", "theory_version", "train_ml_model"]:
            continue
        elif pd.isna(row[col]):
            python_command += f" --{col} ''"
        elif col in ["overwrite", "not_load_model"]:
            if row[col]: python_command += f" --{col}"
        elif col in ["data_mode"]:
            python_command += f" --{col} train"
        elif isinstance(row[col], str) and "&" in row[col]:
            python_command += f" --{col} {' '.join(row[col].split('&'))}"
        else:
            python_command += f" --{col} {row[col]}"
    return python_command

def process_single_eval(row):
    new_row = deepcopy(row)
    new_row["prompt_name"] = "single"
    new_row["eval_mode"] = "pairwise"
    new_row["not_load_model"] = True
    new_row["wandb"] = True
    new_row["n_pairs"] = 2000
    new_row["overwrite"] = False
    return new_row

run_script = ""
for index, row in configs.iterrows():
    n_gpus = row["gpu_request"]
    job_name = f"{row['experiment']}_job_{index}"
    output_file = f"{root_dir}/submit/log_slurm/{job_name}.out"
    error_file = f"{root_dir}/submit/log_slurm/{job_name}.err"

    python_command = create_python_eval_command(row)
    new_row = row.to_dict()

    if row["experiment"] == "ft":
        python_command = create_python_finetune_command(row)

        new_row["peft_variant"] = get_peft_variant_name(
            description="" if pd.isna(row["description"]) else row["description"],
            context="",
            dataset_name=row["dataset_name"].split("&"),
            model_name=row["model_name"],
            eval_mode=row["eval_mode"],
            prompt_name=row["prompt_name"],
            n_demos=row["n_demos"],
            data_mode="train",
        )
        
        for dataset in ["ours_v4", "relca_v2"]:
            new_row["dataset_name"] = dataset
            new_row["n_pairs"] = 2000
            eval_command = create_python_eval_command(new_row)
            python_command += f"\n{eval_command}"

            if row["eval_mode"] == "single":
                wandb_row = process_single_eval(new_row)
                eval_command = create_python_eval_command(wandb_row)
                python_command += f"\n{eval_command}"


    elif row["eval_mode"] == "single":
        wandb_row = process_single_eval(row)
        python_command += f"\n{create_python_eval_command(wandb_row)}"

    

    # create a slurm script
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={output_file}
#SBATCH --error={error_file}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:{n_gpus}
#SBATCH --time=24:00:00
#SBATCH --mem=64G

# Load any necessary modules or activate virtual environment here
source /fsx-project/yuchenzeng/anaconda3/bin/activate meme
export PYTHONPATH=/fsx-project/yuchenzeng/anaconda3/envs/meme/lib/python3.10/site-packages

# Run the Python script
cd ../..

{python_command}
    """

    # write the slurm script to a file
    with open(f"{root_dir}/submit/auto/{job_name}.sh", "w") as f:
        f.write(slurm_script)

    run_script += f"sbatch {job_name}.sh\n"

with open(f"{root_dir}/submit/auto/run_all.sh", "w") as f:
    f.write(run_script)

