from load_dataset import load_dataset
from load_model import load_model
import os, wandb, argparse, pdb
root_dir = os.path.dirname(__file__)
from helper import save_json, read_json, print_configs, set_seed, get_image_size, score_meme_based_on_theory
from configs import support_llms, support_datasets, prompt_processor, image_size_threshold, eval_modes

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
    context = "",
    example = False,
    result_dir = None,
    overwrite = False,
):
    if prompt_name == "cot":
        output_1 = call_model(
            prompt[0], 
            image_paths, 
            max_new_tokens=max_intermediate_tokens,
            save_history=True,
            description=description,
            context=context,
        )
        output_2 = call_model(
            prompt[1], 
            [], 
            max_new_tokens=max_new_tokens,
            history=output_1['history'],
            save_history=True,
            description=description,
            context=context,
        )
        output_dict = {
            'output': output_2['output'],
            'analysis': output_1['output'] + output_2['output'],
        }
    elif prompt_name == "standard":
        output_dict_all = call_model(
            prompt, 
            image_paths, 
            max_new_tokens=max_new_tokens,
            description=description,
            context=context,
        )
        output_dict = {
            'output': output_dict_all['output'],
        }
    elif prompt_name == "theory":
        output_dict = score_meme_based_on_theory(
            meme_path = image_paths[0],
            call_model = call_model,
            result_dir = result_dir,
            max_intermediate_tokens = max_intermediate_tokens,
            max_new_tokens = max_new_tokens,
            example = example,
            description = description,
            context = context,
            overwrite = overwrite,
        )
    else:
        raise ValueError(f"Prompt name {prompt_name} not supported")
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
    context = "",
    max_new_tokens = 1000,
    example = False,
):    
    
    set_seed(seed)
    dataset = load_dataset(dataset_name, binary_classification=True, description=description or context, eval_mode=eval_mode)
    metric = support_datasets[dataset_name]["metric"]
    sampled_datasets = []
    
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

    call_model = load_model(model_name, api_key=api_key)

    if eval_mode not in support_datasets[dataset_name]["eval_mode"]:
        raise ValueError(f'Eval mode {eval_mode} not supported by {dataset_name}, please choose from {support_datasets[dataset_name]["eval_mode"]}')
    if eval_mode not in prompt_processor[model_name][metric]:
        raise ValueError(f'Eval mode {eval_mode} not supported, please choose from {list(prompt_processor[model_name][metric].keys())}')
    if prompt_name not in prompt_processor[model_name][metric][eval_mode] and prompt_name != "theory":
        raise ValueError(f'Prompt name {prompt_name} not supported, please choose from {list(prompt_processor[model_name][metric][eval_mode].keys())}')

    if prompt_name == "theory":
        prompt = None
    else:
        prompt = prompt_processor[model_name][metric][eval_mode][prompt_name]['prompt']

    if description:
        folder_name = f'description_{description}'
    elif context:
        folder_name = f'context_{context}'
    else:
        folder_name = 'multimodal'
    result_dir = f'{root_dir}/results/evaluation/{dataset_name}/{model_name}/{folder_name}/{eval_mode}_{prompt_name}'
    os.makedirs(result_dir, exist_ok=True)

    if eval_mode == 'single':
        corr = 0
        tqdm_bar = tqdm(range(len(dataset)))
        for i in tqdm_bar:
            if context:
                file_path = {
                    "image_path": dataset.loc[i, 'image_path'],
                    "description_path": dataset.loc[i, 'description_path'],
                }
            elif description:
                file_path = dataset.loc[i, 'description_path']
            else:
                file_path = dataset.loc[i, 'image_path']

            file_name = dataset.loc[i, 'image_path'].split('/')[-1].split('.')[0]
            label = dataset.loc[i, 'label']
            result_file = f'{result_dir}/{file_name}.json'

            read_result = False
            if os.path.exists(result_file) and not overwrite and prompt_name != "theory":
                try:
                    result = read_json(result_file)
                    read_result = True
                except KeyboardInterrupt:
                    raise KeyboardInterrupt
                except:
                    pass
            
            if not read_result:
                output_dict = get_output(
                    call_model, 
                    prompt_name,
                    prompt,
                    [file_path], 
                    max_new_tokens=1,
                    description=description,
                    max_intermediate_tokens=max_new_tokens,
                    context=context,
                    example = example,
                    result_dir = result_dir,
                    overwrite = overwrite,
                )

                pred_label = prompt_processor[model_name][metric][eval_mode][prompt_name]['output_processor'](output_dict['output'])

                result = {
                    'file_path': file_path,
                    'label': int(label),  # Convert numpy.int64 to Python int
                    'output_dict': output_dict,
                    'pred_label': int(pred_label),  # Convert numpy.int64 to Python int
                }
                save_json(result, result_file)

            if result['pred_label'] == label: corr += 1
            acc = corr / (i + 1)
            tqdm_bar.set_description(f"Acc: {acc:.4f}")
            if log_wandb:
                wandb.log({'accuracy': acc})

    elif eval_mode == 'pairwise':
        calibrated_corr, corr, consistency = 0, 0, 0
        funny_data = dataset[dataset['label'] == 1].reset_index(drop=True)
        not_funny_data = dataset[dataset['label'] == 0].reset_index(drop=True)

        all_pairs_idx = list(product(range(len(funny_data)), range(len(not_funny_data))))
        # Shuffle the pairs
        random.shuffle(all_pairs_idx)

        tqdm_bar, idx = tqdm(all_pairs_idx), 0
        for i, j in tqdm_bar:
            if n_pairs >=0 and idx >= n_pairs: break

            funny_image_path = funny_data.loc[i, 'image_path']
            not_funny_image_path = not_funny_data.loc[j, 'image_path']
            funny_image_size = get_image_size(funny_image_path)
            not_funny_image_size = get_image_size(not_funny_image_path)
            if funny_image_size > image_size_threshold or not_funny_image_size > image_size_threshold:
                print(f'Image size of {funny_image_path} or {not_funny_image_path} is too large, skip.')
                continue
            else:
                idx += 1

            if description:
                funny_path = funny_data.loc[i, 'description_path']
                not_funny_path = not_funny_data.loc[j, 'description_path']
            elif context:
                funny_path = {
                    "image_path": funny_image_path,
                    "description_path": funny_data.loc[i, 'description_path'],
                }
                not_funny_path = {
                    "image_path": not_funny_image_path,
                    "description_path": not_funny_data.loc[j, 'description_path'],
                }
            else:
                funny_path = funny_image_path
                not_funny_path = not_funny_image_path

            funny_file_name = funny_image_path.split("/")[-1].split(".")[0]
            not_funny_file_name = not_funny_image_path.split("/")[-1].split(".")[0]

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
                if prompt_name != "theory":
                    compare_output_dict_1 = get_output(
                        call_model, 
                        prompt_name,
                        prompt,
                        [funny_path, not_funny_path], 
                        max_new_tokens=1,
                        description=description,
                        max_intermediate_tokens=max_new_tokens,
                        context=context,
                        example = example,
                        result_dir = result_dir,
                        overwrite = overwrite,
                    )
        
                    pred_label_1 = prompt_processor[model_name][metric][eval_mode][prompt_name]['output_processor'](compare_output_dict_1['output'])

                    compare_output_dict_2 = get_output(
                        call_model, 
                        prompt_name,
                        prompt,
                        [not_funny_path, funny_path], 
                        max_new_tokens=1,
                        description=description,
                        max_intermediate_tokens=max_new_tokens,
                        context=context,
                        example = example,
                        result_dir = result_dir,
                        overwrite = overwrite,
                    )
                    pred_label_2 = prompt_processor[model_name][metric][eval_mode][prompt_name]['output_processor'](compare_output_dict_2['output'])

                else:
                    compare_output_dict_1 = get_output(
                        call_model, 
                        prompt_name,
                        prompt,
                        [funny_path], 
                        max_new_tokens=1,
                        description=description,
                        max_intermediate_tokens=max_new_tokens,
                        context=context,
                        example = example,
                        result_dir = result_dir,
                        overwrite = overwrite,
                    )
                    compare_output_dict_2 = get_output(
                        call_model, 
                        prompt_name,
                        prompt,
                        [not_funny_path], 
                        max_new_tokens=1,
                        description=description,
                        max_intermediate_tokens=max_new_tokens,
                        context=context,
                        example = example,
                        result_dir = result_dir,
                        overwrite = overwrite,
                    )

                    pred_label_1 = int(compare_output_dict_1['output'] <= compare_output_dict_2['output'])
                    pred_label_2 = int(compare_output_dict_1['output'] > compare_output_dict_2['output'])
                    
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
            if (result['pred_label_1'] == 0) and (result['pred_label_2'] == 1): 
                calibrated_corr += 1
            if (result['pred_label_1'] == 1 - result['pred_label_2']) and (result['label_1'] in [0,1]): 
                consistency += 1
            if result['pred_label_1'] == 0:
                corr += 1
            if result['pred_label_2'] == 1:
                corr += 1

            acc = corr / idx / 2
            cr = consistency / idx
            calibrated_acc = calibrated_corr / idx
            if log_wandb:
                wandb.log({'accuracy': acc, 'consistency_rate': cr, 'calibrated_accuracy': calibrated_acc})
            tqdm_bar.set_description(f"Acc: {acc:.4f}, CR: {cr:.4f}, Calibrated Acc: {calibrated_acc:.4f}")

    elif eval_mode == 'threeway':
        corr = 0
        meme_options = ["ground_truth", "closest_candidate", "random_candidate"]
        tqdm_bar = tqdm(range(len(dataset)))
        for i in tqdm_bar:
            dataset_i = dataset.loc[i]
            image_path_idxs = random.sample([0, 1, 2], 3)
            post_context = dataset_i["context"]
            image_paths = []

            files_names = []
            for option in meme_options:
                if get_image_size(dataset_i[f"{option}_path"]) > image_size_threshold / 3 * 2:
                    print(f'Image size of {dataset_i[f"{option}_path"]} is too large, skip.')
                    continue
                files_names.append(dataset_i[f"{option}_path"].split("/")[-1].split(".")[0])
            result_name = f'{files_names[0]}_{files_names[1]}_{files_names[2]}'
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
                if description:
                    for meme_option in meme_options:
                        image_paths.append(dataset_i[f"{meme_option}_description_path"])
                elif context:
                    for meme_option in meme_options:
                        image_paths.append({
                            "image_path": dataset_i[f"{meme_option}_path"],
                            "description_path": dataset_i[f"{meme_option}_description_path"],
                        })
                else:
                    for meme_option in meme_options:
                        image_paths.append(dataset_i[f"{meme_option}_path"])

                output_dict = get_output(
                    call_model, 
                    prompt_name,
                    prompt(post_context),
                    [image_paths[idx] for idx in image_path_idxs],
                    max_new_tokens=1,
                    description=description,
                    max_intermediate_tokens=max_new_tokens,
                    context=context,
                    example = example,
                    result_dir = result_dir,
                    overwrite = overwrite,
                )
                pred_label = prompt_processor[model_name][metric][eval_mode][prompt_name]['output_processor'](output_dict['output'])

                result = {
                    "ground_truth_path": dataset_i["ground_truth_path"],
                    "closest_candidate_path": dataset_i["closest_candidate_path"],
                    "random_candidate_path": dataset_i["random_candidate_path"],
                    "context": post_context,
                    "indices": image_path_idxs,
                    "pred_label": pred_label,
                    "output": output_dict['output'],
                }
                
                save_json(result, result_file)

            if result['pred_label'] == image_path_idxs.index(0):
                corr += 1
            acc = corr / (i + 1)
            tqdm_bar.set_description(f"Acc: {acc:.4f}")
            if log_wandb:
                wandb.log({'accuracy': acc})
    return acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    model_names = []
    for model in support_llms:
        model_names.extend(support_llms[model])

    parser.add_argument('--model_name', type=str, default='gpt-4o-mini', choices=model_names)
    parser.add_argument('--dataset_name', type=str, default='ours_v2', choices=list(support_datasets.keys()))
    parser.add_argument('--prompt_name', type=str, default='standard')
    parser.add_argument('--api_key', type=str, default='yz')
    parser.add_argument('--n_per_class', type=int, default=-1, help='-1 for all, otherwise random sample n_per_class for each class')
    parser.add_argument('--n_pairs', type=int, default=-1, help='-1 for all, otherwise random sample n_pairs pairs')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--eval_mode', type=str, default='pairwise', choices=eval_modes)
    parser.add_argument('--description', type=str, default = '')
    parser.add_argument('--context', type=str, default = "")
    parser.add_argument('--max_new_tokens', type=int, default = 1000)
    parser.add_argument('--example', action='store_true')
    args = parser.parse_args()

    print(__file__)
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
        context=args.context,
        max_new_tokens=args.max_new_tokens,
        example = args.example,
    )

    if args.wandb:
        wandb.finish()



