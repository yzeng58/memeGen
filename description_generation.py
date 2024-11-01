import os
from tqdm import tqdm
from load_dataset import load_dataset
from load_model import load_model
from helper import save_json, print_configs, get_image_size
from configs import support_models, support_datasets, description_prompt, image_size_threshold, get_dataset_dir

def generate_dataset_details(
    model_name: str,
    dataset_name: str,
    api_key: str = 'yz',
    overwrite: bool = False,
    prompt_mode: str = 'default',
    max_new_tokens: int = 300,
    shuffle: bool = False,
):
    # Load the dataset
    dataset = load_dataset(
        dataset_name, 
        binary_classification=True,
        description=model_name,
    )
    
    # Load the model
    call_model = load_model(model_name, api_key=api_key)
    
    # Create result directory
    result_dir = f'{get_dataset_dir(dataset_name)}/description/{model_name}'
    os.makedirs(result_dir, exist_ok=True)
    
    # Generate prompt for description
    prompt = description_prompt[prompt_mode]
    
    if shuffle: dataset = dataset.sample(frac=1).reset_index(drop=True)
    
    for i in tqdm(range(len(dataset))):
        image_path = dataset.loc[i, 'image_path']
        meme_size = get_image_size(image_path)
        if meme_size > image_size_threshold:
            print(f"Image size of {os.path.basename(image_path)}: {meme_size}. Skip.")
            continue
        description_path = dataset.loc[i, 'description_path']

        # Check if result already exists
        if os.path.exists(description_path) and not overwrite:
            continue
        
        # Generate description
        description = call_model(prompt, [image_path], max_new_tokens=max_new_tokens)
        
        # Save result
        result = {
            'image_path': image_path,
            'description': description
        }
        save_json(result, description_path)
    
    print(f"Description generation completed for {dataset_name} dataset using {model_name}.")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, choices=[model for models in support_models.values() for model in models])
    parser.add_argument('--dataset_name', type=str, required=True, choices=list(support_datasets.keys()))
    parser.add_argument('--api_key', type=str, default='yz')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--prompt_mode', type=str, default='default')
    parser.add_argument('--max_new_tokens', type=int, default=1000)
    parser.add_argument('--shuffle', action='store_true')
    args = parser.parse_args()
    
    print(__file__)
    print_configs(args)
    
    generate_dataset_details(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        api_key=args.api_key,
        overwrite=args.overwrite,
        prompt_mode=args.prompt_mode,
        max_new_tokens=args.max_new_tokens,
        shuffle=args.shuffle,
    )