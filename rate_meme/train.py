from load_dataset import load_dataset
from load_model import load_model
import os, pdb
root_dir = os.path.dirname(os.path.dirname(__file__))
from helper import save_json, set_seed, get_image_size
from configs import support_eval_datasets, image_size_threshold

import pandas as pd
from tqdm import tqdm
from itertools import product
import random
from utils.eval_utils import get_output, get_folder_name, get_file_path
from configs import support_ml_models

def get_ml_model(
    model_name = 'decision_tree',
):
    if model_name in support_ml_models:
        return support_ml_models[model_name]()
    else:
        raise ValueError(f'Model {model_name} is not supported!')

def train(
    model_name, 
    dataset_name, 
    call_model,
    dataset,
    seed = 42, 
    overwrite = False,
    description = '',
    context = "",
    max_new_tokens = 1000,
    theory_example = False,
    theory_version = 'v1',
    prompt_name = "theory",
    eval_mode = "pairwise",
    n_demos = 0,
    train_ml_model = "xgboost",
):    
    print("----------------------------------")
    print(f'Training ML model for {dataset_name} with {model_name}...')
    
    if prompt_name != "theory" or eval_mode != "pairwise":
        raise ValueError('Only theory prompt and pairwise evaluation mode are supported!')
    if n_demos != 0:
        raise ValueError('n_demos is not supported for training yet!')
    if support_eval_datasets[dataset_name] is None:
        raise ValueError(f'Dataset {dataset_name} is not supported both train and eval!')
    if not support_eval_datasets[dataset_name]["train_test_split"]:
        raise ValueError(f'Dataset {dataset_name} is not supported for train-test split!')

    set_seed(seed)

    prompt = None

    folder_name = get_folder_name(description, context)

    result_dir = f'{root_dir}/results/evaluation/{dataset_name}/{model_name}/{folder_name}/{eval_mode}_{prompt_name}/{n_demos}_shot'
    os.makedirs(result_dir, exist_ok=True)

    funny_data = dataset[dataset['label'] == 1].reset_index(drop=True)
    not_funny_data = dataset[dataset['label'] == 0].reset_index(drop=True)

    all_pairs_idx = list(product(range(len(funny_data)), range(len(not_funny_data))))
    random.shuffle(all_pairs_idx)
    
    demonstrations = []

    tqdm_bar, idx = tqdm(all_pairs_idx), 0
    X, y = [], []
    for i, j in tqdm_bar:

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
        result_file = f'{result_dir}/{result_name}.json'

    
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

        X.extend([list(compare_output_dict_1['scores'].values()), list(compare_output_dict_2['scores'].values())])
        y.extend([1, 0])

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
    
    ml_model = get_ml_model(train_ml_model)
    combined = list(zip(X, y))
    random.shuffle(combined)
    X, y = zip(*combined)
    X, y = list(X), list(y)
    ml_model.fit(X, y)
    print("----------------------------------")
    return ml_model

