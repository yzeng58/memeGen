from load_dataset import load_dataset
from load_model import load_model
import os, wandb, argparse, pdb
root_dir = os.path.dirname(__file__)
from helper import save_json, read_json, print_configs, set_seed, get_image_size
from configs import support_models, support_datasets, prompt_processor, image_size_threshold
from environment import WANDB_INFO
import pandas as pd
from tqdm import tqdm
from itertools import product
import random

def get_output(
    call_model, 
    prompt_name,
    prompt,
    image_paths, 
    max_new_tokens=1,
    max_intermediate_tokens=300,
    description = '',
):
    if 'cot' in prompt_name:
        output_1 = call_model(
            prompt[0], 
            image_paths, 
            max_new_tokens=max_intermediate_tokens,
            save_history=True,
            description=description,
        )
        output_2 = call_model(
            prompt[1], 
            [], 
            max_new_tokens=max_new_tokens,
            history=output_1['history'],
            save_history=True,
            description=description,
        )
        output_dict = {
            'output': output_2['output'],
            'analysis': output_1['output'] + output_2['output'],
        }
    else:
        output_dict_all = call_model(
            prompt, 
            image_paths, 
            max_new_tokens=max_new_tokens,
            description=description,
        )
        output_dict = {
            'output': output_dict_all['output'],
        }
    return output_dict

def evaluate(
    model_name, 
    dataset_name, 
    prompt_name = 'yn',
    api_key = 'yz',  
    n_per_class = -1,
    n_pairs = -1,
    seed = 42, 
    log_wandb = False,
    overwrite = False,
    eval_mode = 'single',
    description = '',
    max_new_tokens = 1000,
):
    set_seed(seed)
    dataset = load_dataset(dataset_name, binary_classification=True, description=description)
    sampled_datasets = []
    for label in dataset['label'].unique():
        label_dataset = dataset[dataset['label'] == label]
        if n_per_class == -1:
            sampled_label_dataset = label_dataset
        else:
            if len(label_dataset) < n_per_class:
                raise ValueError(f"Dataset {dataset_name} does not have enough samples for label {label}")
            sampled_label_dataset = label_dataset.sample(n=n_per_class, random_state=seed, replace=False)
        sampled_datasets.append(sampled_label_dataset)
    dataset = pd.concat(sampled_datasets, ignore_index=True).reset_index(drop=True)
    dataset = dataset.sample(frac=1, random_state=seed).reset_index(drop=True)

    call_model = load_model(model_name, api_key=api_key)

    if eval_mode not in prompt_processor[model_name]:
        raise ValueError(f'Eval mode {eval_mode} not supported, please choose from {list(prompt_processor[model_name].keys())}')
    if prompt_name not in prompt_processor[model_name][eval_mode]:
        raise ValueError(f'Prompt name {prompt_name} not supported, please choose from {list(prompt_processor[model_name][eval_mode].keys())}')

    prompt = prompt_processor[model_name][eval_mode][prompt_name]['prompt']

    description_flag = f'description_{description}' if description else 'multimodal'
    result_dir = f'{root_dir}/results/evaluation/{dataset_name}/{model_name}/{description_flag}/{eval_mode}_{prompt_name}'
    os.makedirs(result_dir, exist_ok=True)

    if eval_mode == 'single':
        corr = 0
        tqdm_bar = tqdm(range(len(dataset)))
        for i in tqdm_bar:
            if description:
                file_path = dataset.loc[i, 'description_path']
            else:
                file_path = dataset.loc[i, 'image_path']
            file_name = file_path.split('/')[-1].split('.')[0]
            label = dataset.loc[i, 'label']
            result_file = f'{result_dir}/{file_name}.json'

            if os.path.exists(result_file) and not overwrite:
                try:
                    result = read_json(result_file)
                    continue
                except KeyboardInterrupt:
                    raise KeyboardInterrupt
                except:
                    pass
            
            output_dict = get_output(
                call_model, 
                prompt_name,
                prompt,
                [file_path], 
                max_new_tokens=1,
                description=description,
                max_intermediate_tokens=max_new_tokens,
            )

            pred_label = prompt_processor[model_name][eval_mode][prompt_name]['output_processor'](output_dict['output'])
            if pred_label == label: corr += 1

            result = {
                'file_path': file_path,
                'label': int(label),  # Convert numpy.int64 to Python int
                'output_dict': output_dict,
                'pred_label': int(pred_label),  # Convert numpy.int64 to Python int
            }
            save_json(result, result_file)
            # Update tqdm description with current accuracy
            total_predictions = i + 1  # One prediction per iteration
            current_acc = corr / total_predictions
            tqdm_bar.set_description(f"Acc: {current_acc:.4f}")

        acc = corr / len(dataset)

    elif eval_mode == 'pairwise':
        corr = 0
        funny_data = dataset[dataset['label'] == 1].reset_index(drop=True)
        not_funny_data = dataset[dataset['label'] == 0].reset_index(drop=True)

        all_pairs_idx = list(product(range(len(funny_data)), range(len(not_funny_data))))
        # Shuffle the pairs
        random.shuffle(all_pairs_idx)

        tqdm_bar, idx = tqdm(all_pairs_idx), 0
        for i, j in tqdm_bar:
            if idx >= n_pairs: break
            if description:
                funny_path = funny_data.loc[i, 'description_path']
                not_funny_path = not_funny_data.loc[j, 'description_path']
            else:
                funny_path = funny_data.loc[i, 'image_path']
                not_funny_path = not_funny_data.loc[j, 'image_path']

            funny_image_path = funny_data.loc[i, 'image_path']
            not_funny_image_path = not_funny_data.loc[j, 'image_path']
            funny_image_size = get_image_size(funny_image_path)
            not_funny_image_size = get_image_size(not_funny_image_path)
            if funny_image_size > image_size_threshold or not_funny_image_size > image_size_threshold:
                print(f'Image size of {funny_path} or {not_funny_path} is too large, skip.')
                continue
            else:
                idx += 1

            funny_file_name = funny_path.split("/")[-1].split(".")[0]
            not_funny_file_name = not_funny_path.split("/")[-1].split(".")[0]

            result_name = f'{funny_file_name}_{not_funny_file_name}'
            result_file = f'{result_dir}/{result_name}.json'

            read_result = False
            if os.path.exists(result_file) and not overwrite:
                try:
                    result = read_json(result_file)
                    read_result = True
                except KeyboardInterrupt:
                    raise KeyboardInterrupt
                except:
                    pass
            
            if not read_result:

                compare_output_dict_1 = get_output(
                    call_model, 
                    prompt_name,
                    prompt,
                    [funny_path, not_funny_path], 
                    max_new_tokens=1,
                    description=description,
                    max_intermediate_tokens=max_new_tokens,
                )
                pred_label_1 = prompt_processor[model_name][eval_mode][prompt_name]['output_processor'](compare_output_dict_1['output'])

                compare_output_dict_2 = get_output(
                    call_model, 
                    prompt_name,
                    prompt,
                    [not_funny_path, funny_path], 
                    max_new_tokens=1,
                    description=description,
                    max_intermediate_tokens=max_new_tokens,
                )
                pred_label_2 = prompt_processor[model_name][eval_mode][prompt_name]['output_processor'](compare_output_dict_2['output'])

                
                result = {
                    'funny_image_path': funny_path,
                    'not_funny_image_path': not_funny_path,
                    'label_1': 0,
                    'label_2': 1,
                    'output_1': compare_output_dict_1,
                    'output_2': compare_output_dict_2,
                    'pred_label_1': int(pred_label_1),
                    'pred_label_2': int(pred_label_2),
                }
                save_json(result, result_file)

            # Update tqdm description with current accuracy
            if (result['pred_label_1'] == 0) and (result['pred_label_2'] == 1): corr += 1
            current_acc = corr / idx
            tqdm_bar.set_description(f"Acc: {current_acc:.4f}")
        acc = corr / len(all_pairs_idx)

    print(f'Accuracy: {acc}')
    if log_wandb:
        wandb.log({'accuracy': acc})
    return acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    model_names = []
    for model in support_models:
        model_names.extend(support_models[model])

    parser.add_argument('--model_name', type=str, default='gpt-4o-mini', choices=model_names)
    parser.add_argument('--dataset_name', type=str, default='ours_v2', choices=support_datasets)
    parser.add_argument('--prompt_name', type=str, default='standard')
    parser.add_argument('--api_key', type=str, default='yz')
    parser.add_argument('--n_per_class', type=int, default=-1, help='-1 for all, otherwise random sample n_per_class for each class')
    parser.add_argument('--n_pairs', type=int, default=-1, help='-1 for all, otherwise random sample n_pairs pairs')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--eval_mode', type=str, default='pairwise', choices=['single', 'pairwise'])
    parser.add_argument('--description', type=str, default = '')
    parser.add_argument('--max_new_tokens', type=int, default = 1000)
    args = parser.parse_args()

    print_configs(args)

    if args.wandb:
        wandb.init(
            project = WANDB_INFO['project'],
            entity = WANDB_INFO['entity'],
            config = vars(args),
        )
    
    evaluate(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        prompt_name=args.prompt_name,
        api_key=args.api_key,
        n_per_class=args.n_per_class,
        n_pairs=args.n_pairs,
        seed=args.seed,
        log_wandb=args.wandb,
        overwrite=args.overwrite,
        eval_mode=args.eval_mode,
        description=args.description,
        max_new_tokens=args.max_new_tokens,
    )

    if args.wandb:
        wandb.finish()



