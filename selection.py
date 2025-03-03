import os, argparse
root_dir = os.path.dirname(__file__)
from configs import prompt_processor, support_gen_datasets, support_llms, support_diffusers, summarizer_prompts, system_prompts_default, prompt_processor_default
from load_model import load_model
from load_dataset import load_dataset
from helper import set_seed, save_json, print_configs, retry_if_fail, read_json
from typing import List, Literal
import itertools
from utils.eval_utils import get_output
import pdb

def select(
    gen_llm_name: str = "gemini-1.5-pro",
    dataset_name: str = "ours_gen_v1",
    api_key: str = "yz",
    n_per_topic: int = -1,
    eval_llm_name: str = "Qwen2.5-72B-Instruct",
    description: bool = True,
    data_mode: Literal["both", "train", "test"] = "both",
    eval_llm_peft_variant: str = "",
    dm_name: str = "stable-diffusion-3-medium-diffusers",
    gen_mode: str = "standard",
    eval_mode: str = "pairwise",
    eval_prompt_name: str = "standard",
    system_prompt_name: str = "",
):
    set_seed(42)
    prompt_names = ["standard", "reversal", "benign2", "lot"]
    if dataset_name not in support_gen_datasets:
        raise ValueError(f"Dataset {dataset_name} not supported!")
    if data_mode in ["train", "test"]:
        if not support_gen_datasets[dataset_name]["train_test_split"]:
            raise ValueError(f"Dataset {dataset_name} does not support train/test split!")
    if eval_mode not in ["pairwise"]:
        raise ValueError(f"Select mode {eval_mode} not supported! Please choose from ['pairwise']")
    
    dataset = load_dataset(dataset_name)
    if data_mode in ["train", "test"]: dataset = dataset[data_mode]

    call_eval_llm_path = f"{eval_llm_name}/{eval_llm_peft_variant}" if eval_llm_peft_variant else f"{eval_llm_name}/pretrained"
    
    call_eval_llm = load_model(call_eval_llm_path, api_key)

    result_dirs = {
        prompt_name: f"{root_dir}/results/generation/{dataset_name}/{gen_llm_name}/{dm_name}/{prompt_name}/{gen_mode}/output"
        # prompt_name: f"{root_dir}/resources/datasets/{dataset_name}/images/{gen_llm_name}/{dm_name}/{prompt_name}/{gen_mode}/output"
        for prompt_name in prompt_names
    }

    contents, file_names = [], []
    if support_gen_datasets[dataset_name]["category"]:
        for topic in dataset:
            if n_per_topic == -1:
                iterations = range(len(dataset[topic]))
            else:
                iterations = range(min(n_per_topic, len(dataset[topic])))
    else:
        for i in dataset.index:
            content = dataset.loc[i]
            contents.append(content)
            file_names.append(f"{i+1}")
    
    
    for content, file_name in zip(contents, file_names):
        file_paths = {}

        if eval_mode == "pairwise":
            combinations = list(itertools.combinations(prompt_names, 2))
            
            for prompt_name_1, prompt_name_2 in combinations:
                result = {}
                result_file_path = f"{root_dir}/results/generation/{dataset_name}/{gen_llm_name}/{dm_name}/selective/{call_eval_llm_path}/{gen_mode}/output/{file_name}_{prompt_name_1}_{prompt_name_2}.json"
                
                path1 = f"{result_dirs[prompt_name_1]}/{file_name}.json"
                path2 = f"{result_dirs[prompt_name_2]}/{file_name}.json"
                
                if not os.path.exists(path1) or not os.path.exists(path2):
                    print(path1, path2)
                    raise ValueError(f"Meme {file_name} does not exist! Please make sure to generate the meme using generation.py before running selection.py!")
                if not path1 in file_paths: file_paths[path1] = 0
                if not path2 in file_paths: file_paths[path2] = 0
                
                if not os.path.exists(result_file_path): 
                    compare_output_dict_1 = get_output(
                        call_model=call_eval_llm,
                        prompt_name=eval_prompt_name,
                        prompt=prompt_processor[eval_llm_name]["funniness"][eval_mode][eval_prompt_name]['prompt'],
                        image_paths=[path1, path2],
                        max_new_tokens=1,
                        description="default" if description else "",
                        max_intermediate_tokens=300,
                        context="",
                        example=False,
                        result_dir=None,
                        overwrite=False,
                        system_prompt_name=system_prompt_name,
                    )
                    compare_output_dict_2 = get_output(
                        call_model=call_eval_llm,
                        prompt_name=eval_prompt_name,
                        prompt=prompt_processor[eval_llm_name]["funniness"][eval_mode][eval_prompt_name]['prompt'],
                        image_paths=[path2, path1],
                        max_new_tokens=1,
                        description="default" if description else "",
                        max_intermediate_tokens=300,
                        context="",
                        example=False,
                        result_dir=None,
                        overwrite=False,
                        system_prompt_name=system_prompt_name,
                    )
                else:
                    result = read_json(result_file_path)
                    compare_output_dict_1 = result[f"{prompt_name_1}_{prompt_name_2}"]
                    compare_output_dict_2 = result[f"{prompt_name_2}_{prompt_name_1}"]

                    
                pred_label_1 = prompt_processor[eval_llm_name]["funniness"][eval_mode][eval_prompt_name]['output_processor'](compare_output_dict_1['output'])
                file_paths[[path1, path2][pred_label_1]] += 1

                pred_label_2 = prompt_processor[eval_llm_name]["funniness"][eval_mode][eval_prompt_name]['output_processor'](compare_output_dict_2['output'])
                file_paths[[path2, path1][pred_label_2]] += 1

                if not os.path.exists(result_file_path):
                    result[f"{prompt_name_1}_{prompt_name_2}"] = compare_output_dict_1
                    result[f"{prompt_name_2}_{prompt_name_1}"] = compare_output_dict_2

                    save_json(result, result_file_path)

        elif eval_mode == "single" and eval_prompt_name == "theory":
            file_paths = {}
            for prompt_name in prompt_names:
                result = {}
                meme_path = f"{result_dirs[prompt_name]}/{file_name}.json"
                result_file_path = f"{root_dir}/results/generation/{dataset_name}/{gen_llm_name}/{dm_name}/selective/{call_eval_llm_path}/{gen_mode}/output/{file_name}_{prompt_name}.json"

                if not os.path.exists(meme_path):
                    raise ValueError(f"Meme {file_name} does not exist! Please make sure to generate the meme using generation.py before running selection.py!")

                if not os.path.exists(result_file_path):
                    output_dict = get_output(
                        call_model=call_eval_llm,
                        prompt_name=eval_prompt_name,
                        prompt=prompt_processor[eval_llm_name]["funniness"][eval_mode][eval_prompt_name]['prompt'],
                        image_paths=[meme_path],
                        max_new_tokens=1,
                        max_intermediate_tokens=300,
                        description="default" if description else "",
                        context="",
                        example=False,
                        result_dir=None,
                        overwrite=False,
                        theory_version="v6",
                        demonstrations=[],
                        system_prompt_name=system_prompt_name,
                        prompt_position="default",
                    )
                else:
                    result = read_json(result_file_path)
                    output_dict = result["output_dict"]

                result[prompt_name] = output_dict
                save_json(result, result_file_path)
                
                xgboost_model = read_json(f"{root_dir}/results/generation/xgboost_llm_meme_Qwen2-VL-72B-Instruct.json")
                pred_label = xgboost_model[prompt_name]
                file_paths[prompt_name] += pred_label

        else:
            raise ValueError(f"Select mode {eval_mode} not supported! Please choose from ['pairwise']")
                    
        
        save_json(file_paths, f"{root_dir}/results/generation/{dataset_name}/{gen_llm_name}/{dm_name}/selective/{call_eval_llm_path}/{gen_mode}/output/{file_name}.json")
                    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    llm_names = []
    for llm in support_llms:
        llm_names.extend(support_llms[llm])

    dm_names = []
    for dm in support_diffusers:
        dm_names.extend(support_diffusers[dm])

    parser.add_argument("--gen_llm_name", type=str, default="gemini-1.5-pro", choices=llm_names)
    parser.add_argument("--dataset_name", type=str, default="british_complaints", choices=support_gen_datasets.keys()) #
    parser.add_argument("--api_key", type=str, default="yz") 
    parser.add_argument("--n_per_topic", type=int, default=-1, help="Number of social contents to consider per topic")
    parser.add_argument("--eval_llm_name", type=str, default="Qwen2.5-72B-Instruct", choices=llm_names)
    parser.add_argument("--description", type=bool, default=True)
    parser.add_argument("--data_mode", type=str, default="both", choices=["both", "train", "test"])
    parser.add_argument("--eval_llm_peft_variant", type=str, default="")
    parser.add_argument("--dm_name", type=str, default="stable-diffusion-3-medium-diffusers", choices=dm_names)
    parser.add_argument("--gen_mode", type=str, default="standard", choices=["standard", "selective"])
    parser.add_argument("--eval_mode", type=str, default="pairwise", choices=["pairwise", "single"])
    parser.add_argument("--eval_prompt_name", type=str, default="standard", choices=["standard", "theory"])
    parser.add_argument("--system_prompt_name", type=str, default="default")
    args = parser.parse_args()

    select(
        gen_llm_name=args.gen_llm_name,
        dataset_name=args.dataset_name,
        api_key=args.api_key,
        n_per_topic=args.n_per_topic,
        eval_llm_name=args.eval_llm_name,
        description=args.description,
        data_mode=args.data_mode,
        eval_llm_peft_variant=args.eval_llm_peft_variant,
        dm_name=args.dm_name,
        gen_mode=args.gen_mode,
        eval_mode=args.eval_mode,
        eval_prompt_name=args.eval_prompt_name,
        system_prompt_name=args.system_prompt_name,
    )