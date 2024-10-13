from load_dataset import load_dataset
from load_model import load_model
import os, wandb, argparse
root_dir = os.path.dirname(__file__)
from helper import save_json, read_json, print_configs
from configs import support_models, support_datasets, prompt_processor
from environment import WANDB_INFO
import pandas as pd
from tqdm import tqdm

def evaluate(
    model_name, 
    dataset_name, 
    prompt_name = 'yn',
    api_key = 'yz',  
    n_per_class = 35,
    seed = 42, 
    log_wandb = False,
    overwrite = False,
    eval_mode = 'single',
):
    dataset = load_dataset(dataset_name, binary_classification=True)
    sampled_datasets = []
    for label in dataset['label'].unique():
        label_dataset = dataset[dataset['label'] == label]
        if len(label_dataset) < n_per_class:
            raise ValueError(f"Dataset {dataset_name} does not have enough samples for label {label}")
        sampled_label_dataset = label_dataset.sample(n=n_per_class, random_state=seed, replace=False)
        sampled_datasets.append(sampled_label_dataset)
    dataset = pd.concat(sampled_datasets, ignore_index=True).reset_index(drop=True)
    dataset = dataset.sample(frac=1, random_state=seed).reset_index(drop=True)

    call_model = load_model(model_name, api_key=api_key)

    if eval_mode not in prompt_processor:
        raise ValueError(f'Eval mode {eval_mode} not supported, please choose from {list(prompt_processor.keys())}')
    if prompt_name not in prompt_processor[eval_mode]:
        raise ValueError(f'Prompt name {prompt_name} not supported, please choose from {list(prompt_processor[eval_mode].keys())}')

    prompt = prompt_processor[eval_mode][prompt_name]['prompt']

    result_dir = f'{root_dir}/results/evaluation/{dataset_name}/{model_name}'
    os.makedirs(result_dir, exist_ok=True)

    if eval_mode == 'single':
        corr = 0
        for i in tqdm(range(len(dataset))):
            image_path = dataset.loc[i, 'image_path']
            image_name = image_path.split('/')[-1].split('.')[0]
            label = dataset.loc[i, 'label']

            if os.path.exists(f'{result_dir}/{image_name}.json') and not overwrite:
                try:
                    result = read_json(f'{result_dir}/{image_name}.json')
                    continue
                except KeyboardInterrupt:
                    raise KeyboardInterrupt
                except:
                    pass
            
            output = call_model(prompt, [image_path], max_new_tokens=1)
            pred_label = prompt_processor[eval_mode][prompt_name]['output_processor'](output)
            if pred_label == label: corr += 1

            result = {
                'image_path': image_path,
                'label': int(label),  # Convert numpy.int64 to Python int
                'output': output,
                'pred_label': int(pred_label),  # Convert numpy.int64 to Python int
            }
            save_json(result, f'{result_dir}/{image_name}.json')

        acc = corr / len(dataset)

    elif eval_mode == 'pairwise':
        corr = 0
        funny_data = dataset[dataset['label'] == 1].reset_index(drop=True)
        not_funny_data = dataset[dataset['label'] == 0].reset_index(drop=True)

        for i in tqdm(range(len(funny_data))):
            funny_image_path = funny_data.loc[i, 'image_path']
            not_funny_image_path = not_funny_data.loc[i, 'image_path']

            result_name = f'{funny_image_path.split("/")[-1].split(".")[0]}_{not_funny_image_path.split("/")[-1].split(".")[0]}'

            if os.path.exists(f'{result_dir}/{result_name}.json') and not overwrite:
                try:
                    result = read_json(f'{result_dir}/{result_name}.json')
                    continue
                except KeyboardInterrupt:
                    raise KeyboardInterrupt
                except:
                    pass

            compare_output_1 = call_model(prompt, [funny_image_path, not_funny_image_path], max_new_tokens=1)
            compare_output_2 = call_model(prompt, [not_funny_image_path, funny_image_path], max_new_tokens=1)
            pred_label_1 = prompt_processor[eval_mode][prompt_name]['output_processor'](compare_output_1)
            pred_label_2 = prompt_processor[eval_mode][prompt_name]['output_processor'](compare_output_2)
            if pred_label_1 == 0: corr += 1
            if pred_label_2 == 1: corr += 1

            result = {
                'funny_image_path': funny_image_path,
                'not_funny_image_path': not_funny_image_path,
                'label': 0,
                'output_1': compare_output_1,
                'output_2': compare_output_2,
                'pred_label_1': int(pred_label_1),
                'pred_label_2': int(pred_label_2),
            }
            save_json(result, f'{result_dir}/{result_name}.json')
        acc = corr / (len(funny_data) * 2)

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
    parser.add_argument('--dataset_name', type=str, default='memotion', choices=support_datasets)
    parser.add_argument('--prompt_name', type=str, default='yn')
    parser.add_argument('--api_key', type=str, default='yz')
    parser.add_argument('--n_per_class', type=int, default=35)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--eval_mode', type=str, default='pairwise', choices=['single', 'pairwise'])
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
        seed=args.seed,
        log_wandb=args.wandb,
        overwrite=args.overwrite,
        eval_mode=args.eval_mode,
    )

    if args.wandb:
        wandb.finish()



