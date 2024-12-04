from load_dataset import load_dataset
import os, argparse, pdb
root_dir = os.path.dirname(__file__)

from helper import save_json, read_json, print_configs, set_seed, get_image_size
from configs import support_llms, support_eval_datasets, prompt_processor, image_size_threshold, eval_modes, system_prompts, support_llm_properties, system_prompts_default

from environment import CONDA_PATH
import pandas as pd
from tqdm import tqdm
from itertools import product
import random, warnings, yaml, subprocess
from utils.eval_utils import get_folder_name, get_file_path

def get_data_sample_single(
    file_path,
    label,
    prompt,
    model_name,
    metric,
    eval_mode,
    system_prompt_name,
    prompt_name,
):
    if description:
        data_sample = {
            "conversations": [
                {"from": "human", "value": f"Meme: {read_json(file_path)['description']['output']}\n{prompt}"},
                {"from": "gpt", "value": prompt_processor[model_name][metric][eval_mode][prompt_name]['label_processor'](label)},
            ],
            "system": system_prompts[model_name][system_prompt_name],
        }

    elif context:
        information = read_json(file_path)
        data_sample = {
            "conversations": [
                {"from": "human", "value": f"Meme: {information['description']['output']}\n<image>{prompt}"},
                {"from": "gpt", "value": prompt_processor[model_name][metric][eval_mode][prompt_name]['label_processor'](label)},
            ],
            "images": [
                information['image_path'],
            ],
            "system": system_prompts[model_name][system_prompt_name],
        }
    else:
        data_sample = {
            "conversations": [
                {"from": "human", "value": f"<image>{prompt}"},
                {"from": "gpt", "value": prompt_processor[model_name][metric][eval_mode][prompt_name]['label_processor'](label)},
            ],
            "images": [
                file_path,
            ],
            "system": system_prompts[model_name][system_prompt_name],
        }
    return data_sample

def get_data_sample_pairwise(
    path1,
    path2,
    description,
    context,
    prompt,
    model_name,
    metric,
    eval_mode,
    system_prompt_name,
    prompt_name,
    label,
):
    if description:
        data_sample = {
            "conversations": [
                {"from": "human", "value": f"Meme 1: {read_json(path1)['description']['output']}\nMeme 2: {read_json(path2)['description']['output']}\n{prompt}"},
                {"from": "gpt", "value": prompt_processor[model_name][metric][eval_mode][prompt_name]['label_processor'](label)},
            ],
            "system": system_prompts[model_name][system_prompt_name],
        }
    elif context:
        information_1 = read_json(path1)
        information_2 = read_json(path2)
        data_sample = {
            "conversations": [
                {"from": "human", "value": f"Meme 1: {information_1['description']['output']}\n<image>Meme 2: {information_2['description']['output']}\n<image>{prompt}"},
                {"from": "gpt", "value": prompt_processor[model_name][metric][eval_mode][prompt_name]['label_processor'](label)},
            ],
            "images": [
                information_1['image_path'],
                information_2['image_path'],
            ],
            "system": system_prompts[model_name][system_prompt_name],
        }
    else:
        data_sample = {
            "conversations": [
                {"from": "human", "value": f"<image><image>{prompt}"},
                {"from": "gpt", "value": prompt_processor[model_name][metric][eval_mode][prompt_name]['label_processor'](label)},
            ],
            "images": [
                path1,
                path2,
            ],
            "system": system_prompts[model_name][system_prompt_name],
        }
    return data_sample

def preprocess(
    model_name, 
    dataset_name, 
    prompt_name = 'standard',
    n_per_class = -1,
    n_pairs = -1,
    seed = 42, 
    eval_mode = 'single',
    description = '',
    context = "",
    not_load_model = False,
    ensemble = False,
    n_demos = 0,
    difficulty = 'easy',
    system_prompt_name = "",
    data_mode = "train", # "train" or "test" or "both"
    dataset_save_name = "",
    mix = False,
):            
    if "difficulty" in support_eval_datasets[dataset_name]:
        if difficulty not in support_eval_datasets[dataset_name]["difficulty"]:
            raise ValueError(f'Difficulty {difficulty} not supported for {dataset_name}, please choose from {support_eval_datasets[dataset_name]["difficulty"]}')
        
    if ensemble:
        if len(model_name) <= 1:
            raise ValueError('Ensemble evaluation mode requires multiple model names!')
        if len(description) != len(model_name) or len(context) != len(model_name):
            raise ValueError('Ensemble evaluation mode requires the same number of descriptions and contexts for each model!')
        if prompt_name != "standard" or eval_mode != "pairwise":
            raise ValueError('Ensemble evaluation mode only supports standard prompt and pairwise evaluation mode!')
        if not not_load_model:
            warnings.warn('Ensemble evaluation mode does not support loading models. Setting not_load_model=True.')
            not_load_model = True
        
    if eval_mode not in support_eval_datasets[dataset_name]["eval_mode"]:
        raise ValueError(f'Eval mode {eval_mode} not supported by {dataset_name}, please choose from {support_eval_datasets[dataset_name]["eval_mode"]}')
    if prompt_name not in eval_modes[eval_mode]:
        raise ValueError(f'Prompt name {prompt_name} not supported, please choose from {eval_modes[eval_mode]}')
    if support_eval_datasets[dataset_name] is None:
        raise ValueError(f'Dataset {dataset_name} is not supported for evaluation!')
    
    if n_demos > 0:
        raise ValueError('Demonstrations are not supported yet for data preprocessing!')
        if eval_mode == 'threeway':
            raise ValueError('Demonstrations are not supported in threeway evaluation mode!')
        if prompt_name != 'standard':
            raise ValueError('Demonstrations are only supported in standard prompt!')
        
    if eval_mode == "threeway": raise ValueError('Threeway evaluation mode is not supported yet for data preprocessing!')
    if prompt_name == "theory": raise ValueError('Theory prompt is not supported yet for data preprocessing!')
    
    set_seed(seed)
    dataset = load_dataset(
        dataset_name, 
        binary_classification=True, 
        description=description or context, 
        eval_mode=eval_mode, 
        train_test_split=True,
        difficulty=difficulty,
    )
    if data_mode == "train":
        dataset = dataset['train']
    elif data_mode == "test":
        dataset = dataset['test']


    metric = support_eval_datasets[dataset_name]["metric"]
    sampled_datasets = []
    if (not ensemble) and (eval_mode not in prompt_processor[model_name][metric]):
        raise ValueError(f'Eval mode {eval_mode} not supported, please choose from {list(prompt_processor[model_name][metric].keys())}')


    if n_per_class >= 0:
        if eval_mode == 'threeway':
            raise ValueError('Threeway evaluation mode is not compatible with n_per_class!')
        
        for label in dataset['label'].unique():
            label_dataset = dataset[dataset['label'] == label]

            if len(label_dataset) < n_per_class:
                raise ValueError(f"Dataset {dataset_name} does not have enough samples for label {label}")
            sampled_label_dataset = label_dataset.sample(n=n_per_class, random_state=seed, replace=False)
            sampled_datasets.append(sampled_label_dataset)

        dataset = pd.concat(sampled_datasets, ignore_index=True).reset_index(drop=True)
        dataset = dataset.sample(frac=1, random_state=seed).reset_index(drop=True)

    if prompt_name == "theory" or ensemble:
        # the pipeline is implemented in the score_meme_based_on_theory function
        prompt = None
    else:
        prompt = prompt_processor[model_name][metric][eval_mode][prompt_name]['prompt']


    result_dir = f'{root_dir}/llama_factory/data'
    # result_dir = f'{get_dataset_dir(dataset_name)}/llama_factory/{model_name}/{folder_name}/{eval_mode}_{prompt_name}/{n_demos}_shot'
    os.makedirs(result_dir, exist_ok=True)  

    if eval_mode == 'single':
        if n_demos > 0:
            demonstration_idxs = []
            n_class = len(dataset['label'].unique())
            n_per_class = n_demos // n_class
            remaining = n_demos % n_class
            for i, label in enumerate(dataset['label'].unique()):
                label_dataset = dataset[dataset['label'] == label]
                # Add one extra example to some classes if n_demos doesn't divide evenly
                n_samples = n_per_class + (1 if i < remaining else 0)
                available_idxs = label_dataset.index.tolist()
                selected_idxs = []
                
                while len(selected_idxs) < n_samples and available_idxs:
                    # Sample one index
                    idx = random.choice(available_idxs)
                    available_idxs.remove(idx)
                    
                    # Check its size
                    file_path = get_file_path(dataset, context, description, idx)
                    if os.path.getsize(file_path) < image_size_threshold * 2 / (n_demos + 1):
                        selected_idxs.append(idx)
                
                if len(selected_idxs) < n_samples:
                    raise ValueError(f"Not enough valid examples under threshold for label {label}")
                    
                demonstration_idxs.extend(selected_idxs)
            random.shuffle(demonstration_idxs)

            demonstrations = []
            for idx in demonstration_idxs:
                demonstrations.append({
                    "image_paths": [get_file_path(dataset, context, description, idx)],
                    "label": prompt_processor[model_name][metric][eval_mode][prompt_name]['label_processor'](dataset.loc[idx, 'label']),
                })
        else:
            demonstrations = []

        tqdm_bar = tqdm(range(len(dataset)))
        
        llm_dataset = []
        for i in tqdm_bar:
            file_path = get_file_path(
                dataset = dataset,
                context = context,
                description = description,
                idx = i,
            )
            label = dataset.loc[i, 'label']

            llm_dataset.append(get_data_sample_single(
                file_path=file_path,
                label=label,
                prompt=prompt,
                model_name=model_name,
                metric=metric,
                eval_mode=eval_mode,
                system_prompt_name=system_prompt_name,
                prompt_name=prompt_name,
            ))

    elif eval_mode == 'pairwise':
        funny_data = dataset[dataset['label'] == 1].reset_index(drop=True)
        not_funny_data = dataset[dataset['label'] == 0].reset_index(drop=True)

        all_pairs_idx = list(product(range(len(funny_data)), range(len(not_funny_data))))
        random.shuffle(all_pairs_idx)

        if n_demos > 0:
            demonstration_idxs = []
            remaining_pairs = all_pairs_idx.copy()
            while len(demonstration_idxs) < n_demos and remaining_pairs:
                pair_idx = random.choice(remaining_pairs)
                remaining_pairs.remove(pair_idx)
                i, j = pair_idx
                funny_image_size = get_image_size(funny_data.loc[i, 'image_path'])
                not_funny_image_size = get_image_size(not_funny_data.loc[j, 'image_path'])
                if (funny_image_size <= (image_size_threshold / n_demos)) and (not_funny_image_size <= (image_size_threshold / n_demos)):
                    demonstration_idxs.append(pair_idx)
            if len(demonstration_idxs) < n_demos:
                raise ValueError(f'Could only find {len(demonstration_idxs)} valid demonstration pairs out of requested {n_demos}')

            n_true = n_demos // 2
            n_false = n_demos - n_true
            demonstration_labels = [0] * n_true + [1] * n_false
            random.shuffle(demonstration_labels)

            if random.choice([True, False]): demonstration_labels = [1 - label for label in demonstration_labels]

            demonstrations = []
            for idx, (demonstration_idx, demonstration_label) in enumerate(zip(demonstration_idxs, demonstration_labels)):
                images_paths = [
                    get_file_path(funny_data, context, description, demonstration_idx[0]), 
                    get_file_path(not_funny_data, context, description, demonstration_idx[1])
                ]
                if demonstration_label: images_paths = images_paths[::-1]

                demonstrations.append({
                    "image_paths": images_paths,
                    "label": prompt_processor[model_name][metric][eval_mode][prompt_name]['label_processor'](demonstration_label),
                })

        else:
            demonstrations = []

        tqdm_bar, idx = tqdm(all_pairs_idx), 0
        llm_dataset = []
        for i, j in tqdm_bar:
            if n_pairs >=0 and idx >= n_pairs: break

            funny_image_path = funny_data.loc[i, 'image_path']
            not_funny_image_path = not_funny_data.loc[j, 'image_path']
            funny_image_size = get_image_size(funny_image_path)
            not_funny_image_size = get_image_size(not_funny_image_path)
            if funny_image_size > image_size_threshold or not_funny_image_size > image_size_threshold:
                print(f'Image size of {funny_image_path}({funny_image_size}) or {not_funny_image_path}({not_funny_image_size}) is too large, skip.')
                continue
            else:
                idx += 1

            funny_path = get_file_path(funny_data, context, description, i)
            not_funny_path = get_file_path(not_funny_data, context, description, j)


            if not prompt_name in ["theory", "single"]:
                llm_dataset.append(get_data_sample_pairwise(
                    path1=funny_path,
                    path2=not_funny_path,
                    description=description,
                    context=context,
                    prompt=prompt,
                    model_name=model_name,
                    metric=metric,
                    eval_mode=eval_mode,
                    system_prompt_name=system_prompt_name,
                    prompt_name=prompt_name,
                    label=0,
                ))
                llm_dataset.append(get_data_sample_pairwise(
                    path1=not_funny_path,
                    path2=funny_path,
                    description=description,
                    context=context,
                    prompt=prompt,
                    model_name=model_name,
                    metric=metric,
                    eval_mode=eval_mode,
                    system_prompt_name=system_prompt_name,
                    prompt_name=prompt_name,
                    label=1,
                ))

    file_name = f"{dataset_save_name}.json"
    file_path = f'{result_dir}/{file_name}'
    if os.path.exists(file_path) and mix: 
        llm_dataset.extend(read_json(file_path))
        random.shuffle(llm_dataset)
    save_json(llm_dataset, file_path)
    print(f"Saved {len(llm_dataset)} samples to {result_dir}/{file_name}")
    dataset_info = read_json(f'{root_dir}/llama_factory/data/dataset_info.json')
    if dataset_save_name not in dataset_info:
        columns = {"messages": "conversations","chosen": "chosen"}
        if not description:
            columns["images"] = "images"

        dataset_info[dataset_save_name] = {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": columns
        }
        save_json(dataset_info, f'{root_dir}/llama_factory/data/dataset_info.json')

def finetune(
    model_name, 
    dataset_name, 
    prompt_name, 
    n_demos, 
    seed, 
    eval_mode, 
    description, 
    context, 
    not_load_model, 
    ensemble,  
    difficulty, 
    system_prompt_name, 
    data_mode,
    n_per_class,
    n_pairs,
):
    modality_mode = get_folder_name(description, context)

    datasets = dataset_name
    if len(dataset_name) > 1: 
        dataset_name = '_mix_'.join(datasets)
    else:
        dataset_name = dataset_name[0]
        
    dataset_save_name = f"{dataset_name}_{model_name}_{modality_mode}_{eval_mode}_{prompt_name}_{n_demos}_shot_{data_mode}"

    print(f"| Preprocessing -- Dataset: {dataset_save_name}")
    for idx, dataset in enumerate(datasets):
        mix = idx > 0

        preprocess(
            model_name=model_name,
            dataset_name=dataset,
            prompt_name=prompt_name,
            n_per_class=n_per_class,
            n_pairs=n_pairs,
            seed=seed,
            eval_mode=eval_mode,
            description=description,
            context=context,
            not_load_model=not_load_model,
            ensemble=ensemble,
            n_demos=n_demos,
            difficulty=difficulty,
            system_prompt_name=system_prompt_name,
            data_mode=data_mode,
            dataset_save_name=dataset_save_name,
            mix = mix
        )

    finetune_config = {
        "model_name_or_path": support_llm_properties[model_name]['huggingface_repo_name'],
        "quantization_bit": 4,
        "quantization_method": "bitsandbytes",
        
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora", 
        "lora_target": "all",
        
        "dataset": dataset_save_name,
        "template": support_llm_properties[model_name]['chat_template'],
        "cutoff_len": 2048,
        "max_samples": 1000,
        "overwrite_cache": True,
        "preprocessing_num_workers": 16,
        
        "output_dir": f"saves/{model_name}/{dataset_save_name}",
        "logging_steps": 10,
        "save_steps": 500,
        "plot_loss": True,
        "overwrite_output_dir": True,
        
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 1.0e-4,
        "num_train_epochs": 3.0,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "bf16": True,
        "ddp_timeout": 180000000,
        
        "val_size": 0.1,
        "per_device_eval_batch_size": 1,
        "eval_strategy": "steps",
        "eval_steps": 500
    }
    
    finetune_config_path = f"{root_dir}/llama_factory/configs/{dataset_save_name}_finetune.yaml"
    print(f"| Getting Fine-Tuning Configs -- Saving yaml config to {finetune_config_path}")
    with open(finetune_config_path, 'w') as f:
        yaml.dump(finetune_config, f)

    finetune_cmd = f"{CONDA_PATH} run -n meme llamafactory-cli train {finetune_config_path}"
    print(f"| Starting Fine-Tuning -- Running command: {finetune_cmd}")
    subprocess.run(
        finetune_cmd,
        shell=True,
        cwd=f"{root_dir}/llama_factory",
        check=True,
        stdout=None,
        stderr=None,
        bufsize=1,
        universal_newlines=True
    )

    merge_config = {
        "model_name_or_path": support_llm_properties[model_name]["huggingface_repo_name"],
        "adapter_name_or_path": f"saves/{model_name}/qlora_{dataset_save_name}",
        "template": support_llm_properties[model_name]["chat_template"],
        "finetuning_type": "lora",
        
        "export_dir": f"../models/{model_name}/qlora_{dataset_save_name}",
        "export_size": 2,
        "export_device": "cpu",
        "export_legacy_format": False
    }
    merge_config_path = f"{root_dir}/llama_factory/configs/{dataset_save_name}_merge.yaml"
    print(f"| Getting Merge Configs -- Saving yaml config to {merge_config_path}")
    with open(merge_config_path, 'w') as f:
        yaml.dump(merge_config, f)  

    merge_cmd = f"{CONDA_PATH} run -n meme llamafactory-cli export {merge_config_path}"
    print(f"| Starting Merge -- Running command: {merge_cmd}")
    subprocess.run(
        merge_cmd,
        shell=True,
        cwd=f"{root_dir}/llama_factory",
        check=True,
        stdout=None,
        stderr=None,
        bufsize=1,
        universal_newlines=True
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    model_names = []
    for model in support_llms:
        model_names.extend(support_llms[model])

    parser.add_argument('--model_name', type=str, nargs='+', default=['Qwen2-VL-2B-Instruct'], choices=model_names)
    parser.add_argument('--dataset_name', type=str, nargs='+', default=['relca'], choices=list(support_eval_datasets.keys()))
    parser.add_argument('--prompt_name', type=str, default='standard')
    parser.add_argument('--n_demos', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_mode', type=str, default='pairwise', choices=list(eval_modes.keys()))
    parser.add_argument('--description', type=str, nargs='+', default = [''])
    parser.add_argument('--context', type=str, nargs='+', default = [""])
    parser.add_argument('--not_load_model', action='store_true')
    parser.add_argument('--ensemble', action='store_true')
    parser.add_argument('--difficulty', type=str, default='easy', choices=['easy', 'medium'])
    parser.add_argument('--system_prompt_name', type=str, default='evaluator', choices=list(system_prompts_default.keys()))
    parser.add_argument('--data_mode', type=str, default='train', choices=['train', 'test', 'both'])
    parser.add_argument('--n_per_class', type=int, default=-1, help='-1 for all, otherwise random sample n_per_class for each class')
    parser.add_argument('--n_pairs', type=int, default=5000, help='-1 for all, otherwise random sample n_pairs pairs')
    args = parser.parse_args()

    print(__file__)
    print_configs(args)

    if args.ensemble:
        model_name = args.model_name
        description = args.description
        context = args.context
    else:
        model_name = args.model_name[0]
        description = args.description[0]
        context = args.context[0]
    
    finetune(
        model_name=model_name,
        dataset_name=args.dataset_name,
        prompt_name=args.prompt_name,
        n_demos=args.n_demos,
        seed=args.seed,
        eval_mode=args.eval_mode,
        description=description,
        context=context,
        not_load_model=args.not_load_model,
        ensemble=args.ensemble,
        difficulty=args.difficulty,
        system_prompt_name=args.system_prompt_name,
        data_mode=args.data_mode,
        n_per_class=args.n_per_class,
        n_pairs=args.n_pairs,
    )




