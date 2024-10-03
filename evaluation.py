from load_dataset import load_dataset
from load_model import load_model
import os, wandb, argparse
root_dir = os.path.dirname(__file__)
from helper import save_json, read_json, print_configs
from configs import support_models, support_datasets, prompt_processor
from environment import WANDB_INFO
import pandas as pd

def evaluate(
    model_name, 
    dataset_name, 
    prompt_name = 'yn',
    api_key = 'yz',  
    n_per_class = 50,
    seed = 42, 
    log_wandb = False,
    overwrite = False,
):
    dataset = load_dataset(dataset_name, binary_classification=True)
    sampled_datasets = []
    for label in dataset['label'].unique():
        label_dataset = dataset[dataset['label'] == label]
        sampled_label_dataset = label_dataset.sample(n=n_per_class, random_state=seed)
        sampled_datasets.append(sampled_label_dataset)
    dataset = pd.concat(sampled_datasets, ignore_index=True).reset_index(drop=True)
    dataset = dataset.sample(frac=1, random_state=seed).reset_index(drop=True)

    call_model = load_model(model_name, api_key=api_key)
    prompt = prompt_processor[prompt_name]['prompt']

    result_dir = f'{root_dir}/results/evaluation/{dataset_name}/{model_name}'
    os.makedirs(result_dir, exist_ok=True)

    corr = 0
    for i in range(len(dataset)):
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

        pred_label = prompt_processor[prompt_name]['output_processor'](output)
        if pred_label == label: corr += 1

        result = {
            'image_path': image_path,
            'label': int(label),  # Convert numpy.int64 to Python int
            'output': output,
            'pred_label': int(pred_label),  # Convert numpy.int64 to Python int
        }
        save_json(result, f'{result_dir}/{image_name}.json')

    acc = corr / len(dataset)
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
    parser.add_argument('--prompt_name', type=str, default='yn', choices=list(prompt_processor.keys()))
    parser.add_argument('--api_key', type=str, default='yz')
    parser.add_argument('--n_per_class', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
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
    )

    if args.wandb:
        wandb.finish()



