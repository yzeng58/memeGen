import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from configs import prompt_processor, get_dataset_dir, support_models, support_datasets
from tqdm import tqdm
from load_model import load_model
from load_dataset import load_dataset
from helper import save_json, print_configs
import argparse, pdb
from PIL import Image
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

    return label_1 == 0 and label_2 == 1

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

def get_image_size(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
    return width*height

def classify_memes(
    dataset_name = 'memotion',
    model_name = 'Qwen2-VL-72B-Instruct',
    api_key = "yz",
    description = "",
    overwrite = False,
    funny_anchor = f"{root_dir}/collection/anchors/hilarious.jpg",
    reverse = False,
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
    funny_anchor_name = os.path.basename(funny_anchor).rsplit('.', 1)[0]
    
    if reverse:
        image_paths = dataset['image_path'][::-1]
    else:
        image_paths = dataset['image_path']

    for meme_path in tqdm(image_paths):
        meme_size = get_image_size(meme_path)
        if meme_size > 500000:
            print(f"Image size of {os.path.basename(meme_path)}: {meme_size}. Skip.")
            continue

        meme_name = os.path.basename(meme_path).rsplit('.', 1)[0]
        result_path = f"{result_dir}/{meme_name}_{funny_anchor_name}.json"

        if os.path.exists(result_path) and not overwrite: continue

        funny_label = is_funny(meme_path, call_model)
        universal_label = is_universal(meme_path, call_model)
        toxic_label = is_toxic(meme_path, call_model)
    
        result = {
            'is_funny': funny_label,
            'is_universal': universal_label,
            'is_toxic': toxic_label,
            "image_path": meme_path,
        }
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
    parser.add_argument("--funny_anchor", type=str, default=f"{root_dir}/collection/anchors/hilarious.jpg")
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()

    print_configs(args)

    classify_memes(
        dataset_name = args.dataset_name,
        model_name = args.model_name,
        api_key = args.api_key,
        description = args.description,
        overwrite = args.overwrite,
        funny_anchor = args.funny_anchor,
        reverse = args.reverse,
    )


    

    