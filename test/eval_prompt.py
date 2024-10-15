import os, argparse, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from load_dataset import load_dataset
from load_model import load_model
from helper import print_configs
from configs import support_models, support_datasets, prompt_processor
import pandas as pd
from tqdm import tqdm

def get_output(
    call_model, 
    prompt_name,
    prompt,
    image_paths, 
    max_new_tokens=1,
    max_intermediate_tokens=500,
):
    if prompt_name == 'cot':
        output_dict_1 = call_model(
            prompt[0], 
            image_paths, 
            max_new_tokens=max_intermediate_tokens,
            save_history=True,
        )
        print("======================================================")
        print('First step output:')
        print(output_dict_1['output'])
        output_dict = call_model(
            prompt[1], 
            [], 
            max_new_tokens=max_new_tokens,
            history=output_dict_1['history'],
            save_history=True,
        )
        print("======================================================")
        print('Second step output:')
        print(output_dict['output'])
        print("======================================================")
    else:
        output_dict = call_model(
            prompt, 
            image_paths, 
            max_new_tokens=max_new_tokens,
        )
        print("======================================================")
        print('Output:')
        print(output_dict['output'])
        print("======================================================")
    return output_dict

def evaluate(
    model_name, 
    dataset_name, 
    api_key = 'yz',  
    n_per_class = 35,
    seed = 42, 
    eval_mode = 'single',
    description = '',
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

    if eval_mode not in prompt_processor[model_name]:
        raise ValueError(f'Eval mode {eval_mode} not supported, please choose from {list(prompt_processor[model_name].keys())}')

    
    while True:
        success = False
        while not success:
            prompt_name = input('Please input the prompt name: (normal/cot/quit)')
            if prompt_name in ['normal', 'cot']:
                success = True
            elif prompt_name == 'quit':
                exit()
            else:
                print('Invalid prompt name, please try again')

        if prompt_name == 'normal':
            prompt = input('Please input the prompt: ')
        elif prompt_name == 'cot':
            prompt = [
                input('Please input the first step of the prompt: '),
                input('Please input the second step of the prompt: '),
            ]

        result_dir = f'{root_dir}/results/evaluation/{dataset_name}/{model_name}/{eval_mode}_{prompt_name}'
        os.makedirs(result_dir, exist_ok=True)

        if eval_mode == 'single':
            tqdm_bar = tqdm(range(len(dataset)))
            for i in tqdm_bar:
                image_path = dataset.loc[i, 'image_path']
                label = dataset.loc[i, 'label']
                
                print('\n\n')
                print("======================================================")
                print(f'Evaluating {image_path}')
                get_output(
                    call_model, 
                    prompt_name,
                    prompt,
                    [image_path], 
                    max_new_tokens=1,
                )
                print('\n\n')

        elif eval_mode == 'pairwise':
            try:
                funny_data = dataset[dataset['label'] == 1].reset_index(drop=True)
                not_funny_data = dataset[dataset['label'] == 0].reset_index(drop=True)

                tqdm_bar = tqdm(range(len(funny_data)))
                for i in tqdm_bar:
                    funny_image_path = funny_data.loc[i, 'image_path']
                    not_funny_image_path = not_funny_data.loc[i, 'image_path']


                    print('\n\n')
                    print("======================================================")
                    print(f'Evaluating (funny) {funny_image_path} and (not funny) {not_funny_image_path}')    
                    get_output(
                        call_model, 
                        prompt_name,
                        prompt,
                        [funny_image_path, not_funny_image_path], 
                        max_new_tokens=1,
                    )
                    print('\n\n')
                    print("======================================================")
                    print(f'Evaluating (not funny) {not_funny_image_path} and (funny) {funny_image_path}')    
                    get_output(
                        call_model, 
                        prompt_name,
                        prompt,
                        [not_funny_image_path, funny_image_path], 
                        max_new_tokens=1,
                    )
                    print('\n\n')
            except KeyboardInterrupt:
                continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    model_names = []
    for model in support_models:
        model_names.extend(support_models[model])

    parser.add_argument('--model_name', type=str, default='gpt-4o-mini', choices=model_names)
    parser.add_argument('--dataset_name', type=str, default='ours_v2', choices=support_datasets)
    parser.add_argument('--api_key', type=str, default='yz')
    parser.add_argument('--n_per_class', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_mode', type=str, default='pairwise', choices=['single', 'pairwise'])
    parser.add_argument('--description', type=str, default='')
    args = parser.parse_args()

    print_configs(args)
    

    evaluate(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        api_key=args.api_key,
        n_per_class=args.n_per_class,
        seed=args.seed,
        eval_mode=args.eval_mode,
        description=args.description,
    )



