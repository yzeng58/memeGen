import pandas as pd
import os
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

configs = []
experiments = ["baseline", "text-only", "cot", "icl", "score", "ft"]
datasets = ["relca_v2"]
n_demos = {
    "single": [2, 4, 6, 8],
    "pairwise": [0, 2, 4]
}

n_pairs = {
    "single": -1,
    "pairwise": 2000,
}

description = {
    "ours_v4": "Llama-3.2-90B-Vision-Instruct",
    "relca_v2": "gemini-1.5-pro",
}

wandb = {
    "single": False,
    "pairwise": True,
}

gpu_requests = {
    "gpt-4o-mini": 0,
    "gpt-4o": 0,
    "Llama-3.2-11B-Vision-Instruct": 2,
    "Llama-3.2-90B-Vision-Instruct": 4,
    'Qwen2-VL-2B-Instruct': 1,
    'Qwen2-VL-7B-Instruct': 1,
    'Qwen2-VL-72B-Instruct': 4,
    'gemini-1.5-flash': 0,
    'gemini-1.5-pro': 0,
    'pixtral-12b': 2,
    'Qwen2.5-14B-Instruct': 2,
    'Qwen2.5-72B-Instruct': 4,
    "Llama-3.1-8B-Instruct": 1,
    "Llama-3.1-70B-Instruct": 4,
    "Mistral-7B-Instruct-v0.3": 1,
    "Mixtral-8x22B-Instruct-v0.1": 8,
    "Mistral-Large-Instruct-2407": 4,
}

mllms = [
    # "gpt-4o-mini",
    # "gpt-4o",
    "Llama-3.2-11B-Vision-Instruct",
    "Llama-3.2-90B-Vision-Instruct",
    'Qwen2-VL-2B-Instruct',
    'Qwen2-VL-7B-Instruct',
    'Qwen2-VL-72B-Instruct',
    # 'gemini-1.5-flash',
    # 'gemini-1.5-pro',
    'pixtral-12b',
]

llms = [
    'Qwen2.5-14B-Instruct',
    'Qwen2.5-72B-Instruct',
    "Llama-3.1-8B-Instruct",
    "Llama-3.1-70B-Instruct",
    "Mistral-7B-Instruct-v0.3",
    "Mixtral-8x22B-Instruct-v0.1",
    "Mistral-Large-Instruct-2407",
]

good_mllms = [
    # "gpt-4o",
    # "gemini-1.5-pro",
    "Qwen2-VL-72B-Instruct",
    "pixtral-12b",
]

good_llms = [
    'Qwen2.5-72B-Instruct',
    'Llama-3.1-70B-Instruct',
    'Mixtral-8x22B-Instruct-v0.1',
]

default_config = {
    "model_name": "gemini-1.5-pro",
    "dataset_name": "relca_v2",
    "data_mode": "test",
    "eval_mode": "pairwise",
    "n_pairs": -1,
    "n_demos": 0,
    "wandb": False,
    "overwrite": True,
    "gpu_request": gpu_requests["gemini-1.5-pro"],
    "description": "",
    "prompt_name": "standard",
    "theory_version": "v4",
    "train_ml_model": "",
}


# Baseline
for model in mllms:
    for dataset in datasets:
        for eval_mode in ["single", "pairwise"]:
            config = default_config.copy()
            config.update({
                "model_name": model,
                "dataset_name": dataset,
                "data_mode": "test",
                "eval_mode": eval_mode,
                "n_pairs": n_pairs[eval_mode],
                "n_demos": n_demos[eval_mode][0],
                "wandb": wandb[eval_mode],
                "gpu_request": gpu_requests[model],
            })
            configs.append(config)

# Text-only
model_list = good_mllms + llms
for model in model_list:
    for dataset in datasets:
        for eval_mode in ["single", "pairwise"]:
            config = default_config.copy()
            config.update({
                "model_name": model,
                "dataset_name": dataset,
                "data_mode": "test",
                "eval_mode": eval_mode,
                "n_pairs": n_pairs[eval_mode],
                "n_demos": n_demos[eval_mode][0],
                "wandb": wandb[eval_mode],
                "description": description[dataset],
                "gpu_request": gpu_requests[model],
            })
            configs.append(config)

# CoT
model_list = good_mllms + good_llms
for model in model_list:
    for dataset in datasets:
        if model in good_llms:
            additional_config = {"description": description[dataset]}
        else:
            additional_config = {}
        for eval_mode in ["pairwise"]:
            config = default_config.copy()
            config.update({
                "model_name": model,
                "dataset_name": dataset,
                "data_mode": "test",
                "eval_mode": eval_mode,
                "n_pairs": n_pairs[eval_mode],
                "n_demos": n_demos[eval_mode][0],
                "wandb": wandb[eval_mode],
                "prompt_name": "cot",
                "gpu_request": gpu_requests[model],
            })
            config.update(additional_config)
            configs.append(config)

# ICL
model_list = good_llms + good_mllms
for model in model_list:
    for dataset in datasets:
        if model in good_llms:
            additional_config = {"description": description[dataset]}
        else:
            additional_config = {}
        for eval_mode in ["single", "pairwise"]:
            for n_demo in n_demos[eval_mode][1:]:
                config = default_config.copy()
                config.update({
                    "model_name": model,
                    "dataset_name": dataset,
                    "data_mode": "test",
                    "eval_mode": eval_mode,
                    "n_pairs": n_pairs[eval_mode],
                    "n_demos": n_demo,
                    "wandb": wandb[eval_mode],
                    "gpu_request": gpu_requests[model],
                })
                config.update(additional_config)
                configs.append(config)

# Theory
model_list = good_llms + good_mllms
for model in model_list:
    for dataset in datasets:
        if model in good_llms:
            additional_config = {"description": description[dataset]}
        else:
            additional_config = {}
        for eval_mode in ["pairwise"]:
            config = default_config.copy()
            config.update({
                "model_name": model,
                "dataset_name": dataset,
                "data_mode": "both",
                "eval_mode": eval_mode,
                "n_pairs": n_pairs[eval_mode],
                "wandb": wandb[eval_mode],
                "prompt_name": "theory",
                "theory_version": "v4",
                "train_ml_model": "xgboost",
                "gpu_request": gpu_requests[model],
                "description": description[dataset],
            })
            configs.append(config)

configs_df = pd.DataFrame(configs)
configs_df['n_demos'] = configs_df['n_demos'].astype(float).astype(int)


configs_df.to_csv(f"{root_dir}/experiments/configs.csv", index=False)