import pandas as pd
import os
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

configs = []

# all options: 
# ["baseline", "text-only", "cot", "icl", "theory", "ft", "ft_theory", "pairwise_theory", "ft_pairwise_theory"]
experiments = ["ft_pairwise_theory"]

datasets = ["advanced"]
n_demos = {
    "single": [2, 4, 6, 8],
    "pairwise": [0, 2, 4]
}

n_pairs = {
    "single": -1,
    "pairwise": 2000,
}

description = {
    "advanced": "gemini-1.5-pro",
    "basic": "gemini-1.5-pro",
    "llm_meme": "default",
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
    "DeepSeek-R1-Distill-Qwen-32B": 4,
    "DeepSeek-R1-Distill-Llama-70B": 4,
    "gemini-2.0-flash": 0,
    "o1-2024-12-17": 0,
    "o3-mini-2025-01-31": 0,
    "o3-preview-2024-11-20": 0,
}

mllms = [
    "gpt-4o-mini",
    "gpt-4o",
    "Llama-3.2-11B-Vision-Instruct",
    "Llama-3.2-90B-Vision-Instruct",
    'Qwen2-VL-2B-Instruct',
    'Qwen2-VL-7B-Instruct',
    'Qwen2-VL-72B-Instruct',
    'gemini-1.5-flash',
    'gemini-1.5-pro',
    'pixtral-12b',
    'gemini-2.0-flash',
    'o1-2024-12-17',
    # 'o3-mini-2025-01-31',
    # 'o3-preview-2024-11-20',
]

llms = [
    'Qwen2.5-14B-Instruct',
    'Qwen2.5-72B-Instruct',
    "Llama-3.1-8B-Instruct",
    "Llama-3.1-70B-Instruct",
    "Mistral-7B-Instruct-v0.3",
    "Mixtral-8x22B-Instruct-v0.1",
    "Mistral-Large-Instruct-2407",
    "DeepSeek-R1-Distill-Qwen-32B",
    "DeepSeek-R1-Distill-Llama-70B",
]

good_mllms = [
    "gpt-4o",
    "gemini-1.5-pro",
    "Qwen2-VL-72B-Instruct",
]

good_llms = [
    'Qwen2.5-72B-Instruct',
    'Llama-3.1-70B-Instruct',
]

oom_models = ['Mixtral-8x22B-Instruct-v0.1'] # out of memory

default_config = {
    "model_name": "gemini-1.5-pro",
    "dataset_name": "advanced",
    "data_mode": "test",
    "eval_mode": "pairwise",
    "n_pairs": -1,
    "n_demos": 0,
    "wandb": False,
    "overwrite": False,
    "gpu_request": gpu_requests["gemini-1.5-pro"],
    "description": "",
    "prompt_name": "standard",
    "theory_version": "v6",
    "train_ml_model": "",
    "epochs": 3,
    "lr": 0.0001,
    "max_new_tokens": 300,
}


for experiment in experiments:
    if experiment == "baseline":
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
                        "experiment": experiment,
                    })
                    configs.append(config)
    elif experiment == "text-only":
        model_list = good_mllms + llms
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
                        "description": description.get(dataset, "gemini-1.5-pro"),
                        "gpu_request": gpu_requests[model],
                        "experiment": experiment,
                    })
                    configs.append(config)

    elif experiment == "cot":
        # CoT
        for model in ["Qwen2-VL-72B-Instruct", "Llama-3.1-70B-Instruct", "Qwen2.5-72B-Instruct"]: # good_llms + good_mllms:
            for dataset in datasets:
                if model in good_llms:
                    additional_config = {"description": description.get(dataset, "gemini-1.5-pro")}
                else:
                    additional_config = {}
                for eval_mode in ["single"]: # ["single", "pairwise"]:
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
                        "experiment": experiment,
                    })
                    config.update(additional_config)
                    configs.append(config)

    elif experiment == "icl":
        # ICL
        model_list = good_mllms + good_llms
        for model in model_list:
            for dataset in datasets:
                if model in good_llms:
                    additional_config = {"description": description.get(dataset, "gemini-1.5-pro")}
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
                            "experiment": experiment,
                        })
                        config.update(additional_config)
                        configs.append(config)

    elif experiment == "theory":
        # Theory
        model_list = good_llms + good_mllms
        for model in model_list:
            for dataset in datasets:
                if model in good_llms:
                    additional_config = {"description": description.get(dataset, "gemini-1.5-pro")}
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
                        "theory_version": "v6",
                        "train_ml_model": "xgboost",
                        "gpu_request": gpu_requests[model],
                        "description": '',
                        "experiment": experiment,
                    })
                    config.update(additional_config)
                    configs.append(config)

    elif experiment == "pairwise_theory":
        # Pairwise Theory
        model_list = good_llms + good_mllms
        for model in model_list:
            for dataset in datasets:
                if model in good_llms:
                    additional_config = {"description": description.get(dataset, "gemini-1.5-pro")}
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
                        "prompt_name": "pairwise_theory",
                        "theory_version": "v6",
                        "gpu_request": gpu_requests[model],
                        "description": '',
                        "experiment": experiment,
                        "max_new_tokens": 1000,
                    })
                    config.update(additional_config)
                    configs.append(config)

    elif experiment == "ft":
        # Fine-tuning
        model_list = good_mllms + good_llms
        for model in model_list:
            if model in oom_models: continue
            for dataset in datasets:
                if model in good_llms:
                    additional_config = {"description": description.get(dataset, "gemini-1.5-pro")}
                else:
                    additional_config = {}
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
                        "experiment": experiment,
                        "lr": 0.001,
                        "epochs": 20,
                    })
                    config.update(additional_config)
                    configs.append(config)

    elif experiment == "ft_theory":
        # Fine-tuning with theory
        # model_list = good_mllms + good_llms
        model_list = ["Qwen2-VL-72B-Instruct", "Llama-3.1-70B-Instruct"]
        for model in model_list:
            if model in oom_models: continue
            for dataset in datasets:
                if model in good_llms:
                    additional_config = {"description": description.get(dataset, "gemini-1.5-pro")}
                else:
                    additional_config = {}
                for eval_mode in ["single"]:
                    config = default_config.copy()
                    config.update({
                        "model_name": model,
                        "dataset_name": dataset,
                        "data_mode": "both",
                        "eval_mode": eval_mode,
                        "n_pairs": n_pairs[eval_mode],
                        "wandb": wandb[eval_mode],
                        "prompt_name": "theory",
                        "theory_version": "v6",
                        "train_ml_model": "xgboost",
                        "gpu_request": gpu_requests[model],
                        "experiment": experiment,
                        "lr": 0.001,
                        "epochs": 20,
                    })
                    config.update(additional_config)
                    configs.append(config)

    elif experiment == "ft_pairwise_theory":
        # Fine-tuning with pairwise theory
        model_list = ["Qwen2-VL-72B-Instruct", "Llama-3.1-70B-Instruct"]
        for model in model_list:
            for dataset in datasets:
                if model in good_llms:
                    additional_config = {"description": description.get(dataset, "gemini-1.5-pro")}
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
                        "prompt_name": "pairwise_theory",
                        "theory_version": "v6",
                        "gpu_request": gpu_requests[model],
                        "experiment": experiment,
                        "lr": 0.001,
                        "epochs": 20,
                        "max_new_tokens": 1000,
                    })
                    config.update(additional_config)
                    configs.append(config)

configs_df = pd.DataFrame(configs)
configs_df['n_demos'] = configs_df['n_demos'].astype(float).astype(int)


configs_df.to_csv(f"{root_dir}/experiments/configs.csv", index=False)