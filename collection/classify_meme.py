import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from configs import prompt_processor, get_dataset_dir, support_models, support_datasets, meme_anchors, image_size_threshold
from tqdm import tqdm
from load_model import load_model
from load_dataset import load_dataset
from helper import save_json, print_configs, read_json, get_image_size
import argparse, pdb

def is_funny(
    meme_path, 
    call_model,
    model_name = 'Qwen2-VL-72B-Instruct',
    meme_anchor = f"{root_dir}/collection/anchors/hilarious.jpg",
):
    prompt = prompt_processor[model_name]["pairwise"]["standard"]["prompt"]
    output_1 = call_model(prompt, [meme_path, meme_anchor])['output']
    output_2 = call_model(prompt, [meme_anchor, meme_path])['output']
    label_1 = prompt_processor[model_name]["pairwise"]["standard"]["output_processor"](output_1)
    label_2 = prompt_processor[model_name]["pairwise"]["standard"]["output_processor"](output_2)

    return (label_1 == 0) or (label_2 == 1)

def is_boring(
    meme_path, 
    call_model,
    model_name = 'Qwen2-VL-72B-Instruct',
    meme_anchor = f"{root_dir}/collection/anchors/boring1.jpg",
):
    prompt = prompt_processor[model_name]["pairwise"]["standard"]["prompt"]
    output_1 = call_model(prompt, [meme_path, meme_anchor])['output']
    output_2 = call_model(prompt, [meme_anchor, meme_path])['output']
    label_1 = prompt_processor[model_name]["pairwise"]["standard"]["output_processor"](output_1)
    label_2 = prompt_processor[model_name]["pairwise"]["standard"]["output_processor"](output_2)

    return (label_1 == 1) or (label_2 == 0)

def is_universal(
    meme_path, 
    call_model,
    model_name = 'Qwen2-VL-72B-Instruct',
    region_list = ['China', 'Germany', 'Brazil', 'America'],
):
    prompt = prompt_processor[model_name]["universality"]["prompt"]

    universal_flag = True
    for region in region_list:
        output = call_model(prompt(region), [meme_path])['output']
        label = prompt_processor[model_name]["universality"]["output_processor"](output)
        universal_flag = universal_flag and label == 1

    return universal_flag

def is_toxic(
    meme_path, 
    call_model,
    model_name = 'Qwen2-VL-72B-Instruct',
):
    prompt = prompt_processor[model_name]["toxicity"]["prompt"]
    output = call_model(prompt, [meme_path])['output']
    label = prompt_processor[model_name]["toxicity"]["output_processor"](output)
    return label == 1

def classify_memes(
    dataset_name = 'memotion',
    model_name = 'Qwen2-VL-72B-Instruct',
    api_key = "yz",
    description = "",
    overwrite = False,
    shuffle = False,
):
    if description:
        raise ValueError("Description is not supported for meme collection period.")
    
    result_dir = f"{get_dataset_dir(dataset_name)}/labels/{model_name}"
    os.makedirs(result_dir, exist_ok=True)

    call_model = load_model(
        model_name = model_name, 
        api_key = api_key,
    )

    dataset = load_dataset(dataset_name)

    keys = ['is_hilarious', 'is_funny', 'is_universal', 'is_toxic', "is_boring"]
    
    if shuffle: image_paths = dataset['image_path'].sample(frac=1).values

    for meme_path in tqdm(image_paths):
        meme_size = get_image_size(meme_path)
        if meme_size > image_size_threshold:
            print(f"Image size of {os.path.basename(meme_path)}: {meme_size}. Skip.")
            continue

        meme_name = os.path.basename(meme_path).rsplit('.', 1)[0]
        result_path = f"{result_dir}/{meme_name}.json"

        not_exist_keys, result = [], {}
        if overwrite:
            pass
        elif os.path.exists(result_path):
            result = read_json(result_path)
            for key in keys:
                if not key in result:
                    not_exist_keys.append(key)
                    break
            if len(not_exist_keys) == 0: continue

        for key in not_exist_keys:
            if key == "is_hilarious":
                result[key] = is_funny(meme_path, call_model, meme_anchor = meme_anchors['hilarious'])
            elif key == "is_funny":
                result[key] = is_funny(meme_path, call_model, meme_anchor = meme_anchors['funny'])
            elif key == "is_universal":
                result[key] = is_universal(meme_path, call_model)
            elif key == "is_toxic":
                result[key] = is_toxic(meme_path, call_model)
            elif key == "is_boring":
                boring_label_1 = is_boring(meme_path, call_model, meme_anchor = meme_anchors['boring1'])
                boring_label_2 = is_boring(meme_path, call_model, meme_anchor = meme_anchors['boring2'])
                result[key] = (boring_label_1 == 0) and (boring_label_2 == 1)

        result["image_path"] = meme_path
        save_json(result, result_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    model_names = []
    for model in support_models:
        model_names.extend(support_models[model])

    parser.add_argument("--dataset_name", type=str, default="memotion", choices=support_datasets)
    parser.add_argument("--model_name", type=str, default="Qwen2-VL-72B-Instruct", choices=model_names)
    parser.add_argument("--api_key", type=str, default="yz")
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    args = parser.parse_args()

    print_configs(args)

    classify_memes(
        dataset_name = args.dataset_name,
        model_name = args.model_name,
        api_key = args.api_key,
        description = args.description,
        overwrite = args.overwrite,
        shuffle = args.shuffle,
    )


    

    