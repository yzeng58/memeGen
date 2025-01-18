import pdb
import pandas as pd
import os
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

configs = pd.read_csv(f"{root_dir}/experiments/configs.csv")

def create_python_command(row):
    python_command = "python evaluation.py"
    for col in row.keys():
        if pd.isna(row[col]):
            python_command += f" --{col} ''"
        elif col in ["wandb", "overwrite", "not_load_model"]:
            if row[col]: python_command += f" --{col}"
        elif col == "gpu_request":
            continue
        else:
            python_command += f" --{col} {row[col]}"
    return python_command

for index, row in configs.iterrows():
    n_gpus = row["gpu_request"]
    job_name = f"job_{index}"
    output_file = f"{root_dir}/submit/log_slurm/{job_name}.out"
    error_file = f"{root_dir}/submit/log_slurm/{job_name}.err"

    python_command = create_python_command(row)

    if row["eval_mode"] == "single":
        new_row = row.to_dict()
        new_row["prompt_name"] = "single"
        new_row["eval_mode"] = "pairwise"
        new_row["not_load_model"] = True
        new_row["wandb"] = True
        new_row["n_pairs"] = 2000
        new_row["overwrite"] = False
        python_command += f"\n{create_python_command(new_row)}"

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

run_script = ""
for index in range(len(configs)):
    run_script += f"sbatch job_{index}.sh\n"

with open(f"{root_dir}/submit/auto/run_all.sh", "w") as f:
    f.write(run_script)

