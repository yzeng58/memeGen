from load_dataset import load_dataset
from load_model import load_model
import os, wandb, argparse, pdb
root_dir = os.path.dirname(__file__)
from helper import save_json, read_json, print_configs, set_seed, get_image_size
from rate_meme.rate_meme import score_meme_based_on_theory
from configs import support_llms, support_eval_datasets, prompt_processor, image_size_threshold, eval_modes

from environment import WANDB_INFO_EVAL
import pandas as pd
from tqdm import tqdm
from itertools import product
import random, warnings

def get_file_path(
    dataset,
    context,
    description,
    idx,
):
    if context:
        return {
            "image_path": dataset.loc[idx, 'image_path'],
            "description_path": dataset.loc[idx, 'description_path'],
        }
    elif description:
        return dataset.loc[idx, 'description_path']
    else:
        return dataset.loc[idx, 'image_path']
    
def get_folder_name(
    description,
    context,
):
    if description:
        return f'description_{description}'
    elif context:
        return f'context_{context}'
    else:
        return 'multimodal'

def get_single_output(
    file_path,
    label,
    result_dir,
    overwrite,
    call_model,
    prompt_name,
    prompt,
    description,
    max_new_tokens,
    context,
    example, 
    model_name,
    metric,
    eval_mode,
    theory_version,
    demonstrations = [],
):
    file_name = file_path.split('/')[-1].split('.')[0]
    
    new_result_dir = f"{os.path.dirname(os.path.dirname(result_dir))}/single_standard/{len(demonstrations)}_shot"
    os.makedirs(new_result_dir, exist_ok=True)
    result_file = f'{new_result_dir}/{file_name}.json'

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
        output_dict = get_output(
            call_model, 
            "standard",
            prompt,
            [file_path], 
            max_new_tokens=1,
            description=description,
            max_intermediate_tokens=max_new_tokens,
            context=context,
            example = example,
            result_dir = result_dir,
            overwrite = overwrite,
            theory_version = theory_version,
            demonstrations = demonstrations,
        )

        pred_label = prompt_processor[model_name][metric][eval_mode][prompt_name]['output_processor'](output_dict['output'])

        result = {
            'file_path': file_path,
            'label': int(label),  # Convert numpy.int64 to Python int
            'output_dict': output_dict,
            'pred_label': int(pred_label),  # Convert numpy.int64 to Python int
        }
        save_json(result, result_file)
    return result

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
    theory_version = 'v1',
    demonstrations = [],
):
    if prompt_name == "cot":
        output_1 = call_model(
            prompt[0], 
            image_paths, 
            max_new_tokens=max_intermediate_tokens,
            save_history=True,
            description=description,
            context=context,
            demonstrations = demonstrations,
            system_prompt = 'evaluator',
        )
        output_2 = call_model(
            prompt[1], 
            [], 
            max_new_tokens=max_new_tokens,
            history=output_1['history'],
            save_history=True,
            description=description,
            context=context,
            demonstrations = demonstrations,
            system_prompt = 'evaluator',
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
            demonstrations = demonstrations,
            system_prompt = 'evaluator',
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
            version = theory_version,
            demonstrations = demonstrations,
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
    theory_example = False,
    not_load_model = False,
    theory_version = 'v1',
    ensemble = False,
    n_demos = 0,
):    
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
        if eval_mode == 'threeway':
            raise ValueError('Demonstrations are not supported in threeway evaluation mode!')
        if prompt_name != 'standard':
            raise ValueError('Demonstrations are only supported in standard prompt!')

    set_seed(seed)
    dataset = load_dataset(dataset_name, binary_classification=True, description=description or context, eval_mode=eval_mode)
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

    if not_load_model:
        call_model = None
    else:
        call_model = load_model(model_name, api_key=api_key)

    if prompt_name == "theory" or ensemble:
        # the pipeline is implemented in the score_meme_based_on_theory function
        prompt = None
    else:
        prompt = prompt_processor[model_name][metric][eval_mode][prompt_name]['prompt']


    if ensemble:
        result_dirs = []
        for i, model in enumerate(model_name):
            folder_name = get_folder_name(description[i], context[i])
            result_dir = f'{root_dir}/results/evaluation/{dataset_name}/{model}/{folder_name}/{eval_mode}_{prompt_name}/{n_demos}_shot' 
            result_dirs.append(result_dir)
    else:
        folder_name = get_folder_name(description, context)

        result_dir = f'{root_dir}/results/evaluation/{dataset_name}/{model_name}/{folder_name}/{eval_mode}_{prompt_name}/{n_demos}_shot'
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

        corr = 0
        tqdm_bar = tqdm(range(len(dataset)))
        for i in tqdm_bar:
            file_path = get_file_path(
                dataset = dataset,
                context = context,
                description = description,
                idx = i,
            )
            label = dataset.loc[i, 'label']

            result = get_single_output(
                file_path = file_path,
                label = label,
                result_dir = result_dir,
                overwrite = overwrite,
                call_model = call_model,
                prompt_name = prompt_name,
                prompt = prompt,
                description = description,
                max_new_tokens = max_new_tokens,
                context = context,
                example = theory_example, 
                model_name = model_name,
                metric = metric,
                eval_mode = eval_mode,
                theory_version = theory_version,
                demonstrations = demonstrations,
            )

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

            funny_file_name = funny_image_path.split("/")[-1].split(".")[0]
            not_funny_file_name = not_funny_image_path.split("/")[-1].split(".")[0]

            result_name = f'{funny_file_name}_{not_funny_file_name}'


            read_result = False
            if ensemble:
                best_consistency_idx = []
                for i, result_dir in enumerate(result_dirs):
                    result_file = f'{result_dir}/{result_name}.json'
                    result = read_json(result_file)
                    if result['pred_label_1'] == 1 - result['pred_label_2']:
                        best_consistency_idx.append(i)
                if len(best_consistency_idx) == 0:
                    best_consistency_idx = list(range(len(result_dirs)))

                used_idx = random.choice(best_consistency_idx)
                result = read_json(f'{result_dirs[used_idx]}/{result_name}.json')
                read_result = True
            else:
                result_file = f'{result_dir}/{result_name}.json'

                if os.path.exists(result_file) and not overwrite and not prompt_name == "theory":
                    try:
                        result = read_json(result_file)
                        read_result = True
                    except KeyboardInterrupt:
                        raise KeyboardInterrupt
                    except:
                        pass
            
            if not read_result:
                if not prompt_name in ["theory", "single"]:
                    compare_output_dict_1 = get_output(
                        call_model, 
                        prompt_name,
                        prompt,
                        [funny_path, not_funny_path], 
                        max_new_tokens=1,
                        description=description,
                        max_intermediate_tokens=max_new_tokens,
                        context=context,
                        example = theory_example,
                        result_dir = result_dir,
                        overwrite = overwrite,
                        theory_version = theory_version,
                        demonstrations = demonstrations,
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
                        example = theory_example,
                        result_dir = result_dir,
                        overwrite = overwrite,
                        theory_version = theory_version,
                        demonstrations = demonstrations,
                    )
                    pred_label_2 = prompt_processor[model_name][metric][eval_mode][prompt_name]['output_processor'](compare_output_dict_2['output'])

                elif prompt_name == "theory":
                    compare_output_dict_1 = get_output(
                        call_model, 
                        prompt_name,
                        prompt,
                        [funny_path], 
                        max_new_tokens=1,
                        description=description,
                        max_intermediate_tokens=max_new_tokens,
                        context=context,
                        example = theory_example,
                        result_dir = result_dir,
                        overwrite = overwrite,
                        theory_version = theory_version,
                        demonstrations = demonstrations,
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
                        example = theory_example,
                        result_dir = result_dir,
                        overwrite = overwrite,
                        theory_version = theory_version,
                        demonstrations = demonstrations,
                    )

                    pred_label_1 = int(compare_output_dict_1['output'] <= compare_output_dict_2['output'])
                    pred_label_2 = int(compare_output_dict_1['output'] > compare_output_dict_2['output'])

                elif prompt_name == "single":
                    compare_output_dict_1 = get_single_output(
                        file_path = funny_path,
                        label = 1,
                        result_dir = result_dir,
                        overwrite = overwrite,
                        call_model = call_model,
                        prompt_name = prompt_name,
                        prompt = prompt,
                        description = description,
                        max_new_tokens = max_new_tokens,
                        context = context,
                        example = theory_example,
                        model_name = model_name,
                        metric = metric,
                        eval_mode = eval_mode,
                        theory_version = theory_version,
                        demonstrations = demonstrations,
                    )

                    compare_output_dict_2 = get_single_output(
                        file_path = not_funny_path,
                        label = 0,
                        result_dir = result_dir,
                        overwrite = overwrite,
                        call_model = call_model,
                        prompt_name = prompt_name,
                        prompt = prompt,
                        description = description,
                        max_new_tokens = max_new_tokens,
                        context = context,
                        example = theory_example,
                        model_name = model_name,
                        metric = metric,
                        eval_mode = eval_mode,
                        theory_version = theory_version,
                        demonstrations = demonstrations,
                    )

                    pred_label_1 = int(compare_output_dict_1['pred_label'] <= compare_output_dict_2['pred_label'])
                    pred_label_2 = int(compare_output_dict_1['pred_label'] > compare_output_dict_2['pred_label'])
                else:
                    raise ValueError(f'Prompt name {prompt_name} not supported')
                    
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
            skip_flag = False
            for option in meme_options:
                if get_image_size(dataset_i[f"{option}_path"]) > image_size_threshold / 3 * 2:
                    print(f'Image size of {dataset_i[f"{option}_path"]} is too large, skip.')
                    skip_flag = True
                    continue
                files_names.append(dataset_i[f"{option}_path"].split("/")[-1].split(".")[0])
            if skip_flag: continue
            
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
                    example = theory_example,
                    result_dir = result_dir,
                    overwrite = overwrite,
                    theory_version = theory_version,
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

    parser.add_argument('--model_name', type=str, nargs='+', default=['gemini-1.5-flash'], choices=model_names)
    parser.add_argument('--dataset_name', type=str, default='ours_v2', choices=list(support_eval_datasets.keys()))
    parser.add_argument('--prompt_name', type=str, default='standard')
    parser.add_argument('--api_key', type=str, default='yz')
    parser.add_argument('--n_per_class', type=int, default=-1, help='-1 for all, otherwise random sample n_per_class for each class')
    parser.add_argument('--n_pairs', type=int, default=-1, help='-1 for all, otherwise random sample n_pairs pairs')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--eval_mode', type=str, default='pairwise', choices=list(eval_modes.keys()))
    parser.add_argument('--description', type=str, nargs='+', default = [''])
    parser.add_argument('--context', type=str, nargs='+', default = [""])
    parser.add_argument('--max_new_tokens', type=int, default = 1000)
    parser.add_argument('--theory_example', action='store_true')
    parser.add_argument('--not_load_model', action='store_true', help="Do not load the model. Use this option only when results have already been stored and you want to read the existing results.")
    parser.add_argument('--theory_version', type=str, default='v1', choices=['v1', 'v2'])
    parser.add_argument('--ensemble', action='store_true')
    parser.add_argument('--n_demos', type=int, default=0)
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

    if args.wandb:
        config = {k:v for k,v in vars(args).items()}
        config.update({
            'model_name': model_name,
            'description': description,
            'context': context,
        })
        
        wandb.init(
            project = WANDB_INFO_EVAL['project'],
            entity = WANDB_INFO_EVAL['entity'],
            config = config,
        )
    
    evaluate(
        model_name=model_name,
        dataset_name=args.dataset_name,
        prompt_name=args.prompt_name,
        api_key=args.api_key,
        n_per_class=args.n_per_class,
        n_pairs=args.n_pairs,
        seed=args.seed,
        log_wandb=args.wandb,
        overwrite=args.overwrite,
        eval_mode=args.eval_mode,
        description=description,
        context=context,
        max_new_tokens=args.max_new_tokens,
        theory_example = args.theory_example,
        not_load_model = args.not_load_model,
        theory_version = args.theory_version,
        ensemble = args.ensemble,
        n_demos = args.n_demos,
    )

    if args.wandb:
        wandb.finish()



