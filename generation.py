import os, time, random, itertools, pdb
root_dir = os.path.dirname(__file__)
from configs import prompt_processor, support_gen_datasets, support_llms, support_diffusers, summarizer_prompts, system_prompts_default, prompt_processor_default
from load_model import load_model
from helper import combine_text_and_image, set_seed, save_json, print_configs, retry_if_fail
from rate_meme.rate_meme import score_meme_based_on_theory
from typing import List, Literal
from load_dataset import load_dataset
import argparse, wandb
from environment import WANDB_INFO_GEN
from utils.eval_utils import get_output


@retry_if_fail(max_retries=10, sleep_time=0.1)
def generate_meme_llm(
    call_gen_llm,
    output_path: str,
    gen_llm_name: str = "gpt-4o-mini",
    topic: str = "Working hours are too long",
    prompt_name: str = "standard",
    max_new_tokens: int = 200,
    meme_path: str = None,
    temperature: float = 0.5,
    seed: int = 42,
):
    prompt = prompt_processor[gen_llm_name]["generation"][prompt_name]["prompt"](topic)
    gen_llm_output = call_gen_llm(
        prompt = prompt,
        description = "True",
        max_new_tokens = max_new_tokens,
        temperature = temperature,
        seed = seed,
        system_prompt = "default",
    )["output"]

    gen_llm_output_dict = prompt_processor[gen_llm_name]["generation"][prompt_name]["output_processor"](gen_llm_output)

    output_dict = {
        "image_description": gen_llm_output_dict['image_description'],
        "top_text": gen_llm_output_dict['top_text'],
        "bottom_text": gen_llm_output_dict['bottom_text'],
        "topic": topic,
        "llm_output": gen_llm_output,
        "meme_path": meme_path,
        "description": f"\n* image description: {gen_llm_output_dict['image_description']}\n* top text: {gen_llm_output_dict['top_text']}\n* bottom text: {gen_llm_output_dict['bottom_text']}",
        "output_path": output_path,
    }

    save_json(output_dict, output_path)
    print(f"Output saved to {output_path}")

    return output_dict

def generate_meme_dm(
    call_dm,
    gen_llm_output_dict: dict,
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    negative_prompt: str,
    meme_path: str,
    image_style: str = "cartoon",
    seed: int = 1234,
):
    success_flag = call_dm(
        f"{image_style} style\n{gen_llm_output_dict['image_description']}", 
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        save_path=meme_path,
        seed=seed,
    )

    if success_flag:
        combine_text_and_image(
            meme_path, 
            gen_llm_output_dict['top_text'], 
            gen_llm_output_dict['bottom_text']
        )
        print(f"Meme saved to {meme_path}")
    else:
        raise Exception("Failed to generate image!")
    
def generate_meme_basic(
    result_dir: str,
    call_gen_llm,
    gen_llm_name: str,
    topic: str,
    prompt_name: str,
    max_new_tokens: int,
    short_topic: str,
    description_only: bool = False,
    call_dm = None,
    height: int = 300,
    width: int = 300,
    num_inference_steps: int = 28,
    guidance_scale: float = 7.0,
    negative_prompt: str = "",
    temperature: float = 0.5,
    seed: int = 42,
    image_style: str = "cartoon",
    file_name = None,
): 
    if file_name is None:
        time_string = time.strftime('%Y%m%d_%H%M%S')
        meme_path = f"{result_dir}/meme/{short_topic}_{time_string}.png"
        output_path = f"{result_dir}/output/{short_topic}_{time_string}.json"
    else:
        meme_path = f"{result_dir}/meme/{file_name}.png"
        output_path = f"{result_dir}/output/{file_name}.json"

    os.makedirs(f"{result_dir}/meme", exist_ok=True)
    os.makedirs(f"{result_dir}/output", exist_ok=True)


    gen_llm_output_dict = generate_meme_llm(
        call_gen_llm = call_gen_llm,
        output_path = output_path,
        gen_llm_name = gen_llm_name,
        topic = topic,
        prompt_name = prompt_name,
        max_new_tokens = max_new_tokens,
        meme_path = meme_path,
        temperature = temperature,
        seed = seed,
    )   

    if wandb.run is not None:
        wandb_step_log.update({
            "image_description": gen_llm_output_dict['image_description'],
            "top_text": gen_llm_output_dict['top_text'],
            "bottom_text": gen_llm_output_dict['bottom_text'],
        })

    if not description_only:
        generate_meme_dm(
            call_dm = call_dm,
            gen_llm_output_dict = gen_llm_output_dict,
            height = height,
            width = width,
            num_inference_steps = num_inference_steps,
            guidance_scale = guidance_scale,
            negative_prompt = negative_prompt,
            meme_path = meme_path,
            image_style = image_style,
            seed = seed,
        )

        if wandb.run is not None:
            wandb_step_log.update({
                "meme": wandb.Image(meme_path, caption=file_name),
            })

    if wandb.run is not None:
        wandb.log(wandb_step_log)

    return gen_llm_output_dict

def summarize_topic(
    call_model,
    content, 
    max_new_tokens: int = 10,
    gen_llm_name: str = "gpt-4o-mini",
):
    return call_model(
        prompt = summarizer_prompts[gen_llm_name] + "\n" + content,
        image_paths = [],
        max_new_tokens = max_new_tokens,
        system_prompt = 'default',
        description = "True",
    )

def generate_meme_topic(
    call_gen_llm,
    topic: str = "Working hours are too long",
    description_only: bool = False,
    call_dm = None,
    gen_llm_name: str = "gpt-4o-mini",
    dm_name: str = "stable-diffusion-3-medium-diffusers",
    prompt_name: str = "standard",
    height: int = 280,
    width: int = 280,
    num_inference_steps: int = 28,
    guidance_scale: float = 7.0,
    negative_prompt: str = "",
    seed: int = 1234,
    n_words_in_filename: int = 5,
    max_new_tokens: int = 200,
    n_memes_per_content: int = 1,
    gen_mode: str = "standard",
    n_selected_from: int = 2,
    eval_prompt_name: str = "theory",
    eval_llm_name: str = None,
    call_eval_llm = None,
    temperature: float = 0.5,
    eval_mode: Literal["description", "meme"] = "description",
    theory_version: str = 'v1',
    image_style: str = "cartoon",
    result_dir: str = None,
    file_name: str = None,
    system_prompt_name: str = 'strict_scorer',

): 
    if gen_mode == "selective":
        if eval_llm_name is None:
            raise ValueError("eval_llm_name must be provided if gen_mode is selective!")
        if call_eval_llm is None:
            raise ValueError("call_eval_llm must be provided if gen_mode is selective!")
        if n_selected_from <= 1:
            raise ValueError("n_selected_from must be greater than 1 if gen_mode is selective!")
        if not eval_prompt_name in ["theory", "standard"]:
            raise ValueError(f"eval_prompt_name {eval_prompt_name} not supported!")
        if eval_prompt_name == "standard" and n_selected_from > 3:
            raise ValueError("n_selected_from must be less than or equal to 3 if eval_prompt_name is standard!")
        
    if not description_only and call_dm is None:
        raise ValueError("call_dm must be provided if description_only is False!")

    print(
        f"""
        Generating a meme based on the following topic:

        {topic}
        """
    )

    if wandb.run is not None:
        wandb_step_log.update({
            "topic": topic,
            "file_name": file_name,
        })

    set_seed(seed)
    topic_no_punct = ''.join(c for c in topic if c.isalnum() or c.isspace())
    topic_words = topic_no_punct.split()
    n_words = min(n_words_in_filename, len(topic_words))
    selected_indices = sorted(random.sample(range(len(topic_words)), n_words))
    short_topic_words = [topic_words[i] for i in selected_indices]
    short_topic = "_".join(short_topic_words)

    if result_dir is None:
        result_dir = f"{root_dir}/results/generation/random_attempt/{gen_llm_name}/{dm_name}/{prompt_name}/{gen_mode}"

    results = []

    for i in range(n_memes_per_content):
        seed_iter = random.randint(1, 10000)
        print(f"| Generating meme {i+1}")

        if gen_mode == "standard":
            output_dict = generate_meme_basic(
                result_dir = result_dir,
                file_name = file_name,
                call_gen_llm = call_gen_llm,
                gen_llm_name = gen_llm_name,
                topic = topic,
                prompt_name = prompt_name,
                max_new_tokens = max_new_tokens,
                short_topic = short_topic,
                description_only = description_only,
                call_dm = call_dm,
                height = height,
                width = width,
                num_inference_steps = num_inference_steps,
                guidance_scale = guidance_scale,
                negative_prompt = negative_prompt,
                temperature = temperature,
                seed = seed_iter,
                image_style = image_style,
            )
        elif gen_mode == "selective":
            if result_dir is None:
                output_path = f"{result_dir}/output/final_{short_topic}_{time.strftime('%Y%m%d_%H%M%S')}.json"
            else:
                output_path = f"{result_dir}/output/{file_name}.json"
            
            os.makedirs(f"{result_dir}/output", exist_ok=True)

            output_dict, best_idx = {}, 0
            for j in range(n_selected_from):
                seed_iter = random.randint(1, 10000)
                print(F"| -- For selective generation, generating meme {j+1}")
                gen_llm_output_dict = generate_meme_basic(
                    result_dir = result_dir,
                    file_name = f"{file_name}_{j+1}",
                    call_gen_llm = call_gen_llm,
                    gen_llm_name = gen_llm_name,
                    topic = topic,
                    prompt_name = prompt_name,
                    max_new_tokens = max_new_tokens,
                    short_topic = short_topic,
                    description_only = description_only,
                    call_dm = call_dm,
                    height = height,
                    width = width,
                    num_inference_steps = num_inference_steps,
                    guidance_scale = guidance_scale,
                    negative_prompt = negative_prompt,
                    temperature = temperature,
                    seed = seed_iter,
                    image_style = image_style,
                )

                output_dict[j+1] = {
                    "gen_llm_output": gen_llm_output_dict,
                }

                if eval_prompt_name == "theory":
                    if eval_mode == "description":
                        eval_llm_output_dict = score_meme_based_on_theory(
                            meme_path = gen_llm_output_dict['output_path'],
                            call_model = call_eval_llm,
                            result_dir = result_dir,
                            max_intermediate_tokens = 300,
                            max_new_tokens = 1,
                            example = False,
                            description = gen_llm_output_dict['description'],
                            context = "",
                            version = theory_version,
                            system_prompt_name = system_prompt_name,
                        )
                    elif eval_mode == "meme":
                        eval_llm_output_dict = score_meme_based_on_theory(
                            meme_path = gen_llm_output_dict['meme_path'],
                            call_model = call_eval_llm,
                            result_dir = result_dir,
                            max_intermediate_tokens = 300,
                            max_new_tokens = 1,
                            example = False,
                            description = "",
                            topic = "",
                            theory_version = theory_version,
                        )
                    

                    output_dict[j+1]["eval_llm_output"] = eval_llm_output_dict

                    if eval_llm_output_dict['output'] > output_dict[best_idx]['eval_llm_output']['output']:
                        best_idx = j+1
            
            if eval_prompt_name == "standard":
                if eval_mode == "description":
                    # compare within three
                    # create a list of combination of two of them
                    combinations = list(itertools.combinations(list(range(1, n_selected_from+1)), 2))
                    score = {}
                    for comb in combinations:
                        compare_output_dict = get_output(
                            call_model = call_eval_llm,
                            prompt_name = eval_prompt_name,
                            prompt = prompt_processor[eval_llm_name]["funniness"]["pairwise"][eval_prompt_name]['prompt'],
                            image_paths = [output_dict[comb[0]]['gen_llm_output']['output_path'], output_dict[comb[1]]['gen_llm_output']['output_path']],
                            max_new_tokens = 1,
                            description = gen_llm_output_dict["description"],
                            max_intermediate_tokens = 300,
                            context = "",
                            example = False,
                            result_dir = result_dir,
                            overwrite = False,
                            system_prompt_name = system_prompt_name,
                        )
                        pred_label = prompt_processor[eval_llm_name]["funniness"]["pairwise"][eval_prompt_name]['output_processor'](compare_output_dict['output'])
                        if comb[0] not in score:
                            score[comb[0]] = 0
                        if comb[1] not in score:
                            score[comb[1]] = 0
                        score[comb[pred_label]] += 1
                else:
                    raise ValueError(f"eval_mode {eval_mode} not supported for standard evaluation prompt!")

                best_idx = max(score, key=score.get)

            output_dict["best_idx"] = best_idx
            save_json(output_dict, output_path)

            print(f"Selected meme stored in {output_dict[best_idx]['gen_llm_output']['meme_path']}")
        else:
            raise ValueError(f"gen_mode {gen_mode} not supported!")
        
        results.append(output_dict)
    return results

def generate_meme_content(
    call_gen_llm,
    content: str = "Working hours are too long",
    description_only: bool = False,
    call_dm = None,
    gen_llm_name: str = "gpt-4o-mini",
    dm_name: str = "stable-diffusion-3-medium-diffusers",
    prompt_name: str = "standard",
    height: int = 288,
    width: int = 288,
    num_inference_steps: int = 28,
    guidance_scale: float = 7.0,
    negative_prompt: str = "",
    seed: int = 1234,
    n_words_in_filename: int = 5,
    max_new_tokens: int = 200,
    n_memes_per_content: int = 1,
    gen_mode: str = "standard",
    n_selected_from: int = 2,
    eval_prompt_name: str = "theory",
    eval_llm_name: str = None,
    call_eval_llm = None,
    temperature: float = 0.5,
    eval_mode: Literal["description", "meme"] = "description",
    theory_version: str = 'v1',
    image_style: str = "cartoon",    
    result_dir: str = None,
    file_name: str = None,
    system_prompt_name: str = 'strict_scorer',
):
    topic = summarize_topic(
        call_model = call_gen_llm,
        content = content,
        max_new_tokens = 20,
        gen_llm_name = gen_llm_name,
    )['output']
    if wandb.run is not None:
        wandb_step_log.update({
            "content": content,
        })

    print(f"Summarized topic: {topic}")

    return generate_meme_topic(
        call_gen_llm = call_gen_llm,
        description_only = description_only,
        call_dm = call_dm,
        gen_llm_name = gen_llm_name,
        dm_name = dm_name,
        topic = topic,
        prompt_name = prompt_name,
        height = height,
        width = width,
        num_inference_steps = num_inference_steps,
        guidance_scale = guidance_scale,
        negative_prompt = negative_prompt,
        seed = seed,
        n_words_in_filename = n_words_in_filename,
        max_new_tokens = max_new_tokens,
        n_memes_per_content = n_memes_per_content,
        gen_mode = gen_mode,
        n_selected_from = n_selected_from,
        eval_prompt_name = eval_prompt_name,
        eval_llm_name = eval_llm_name,
        call_eval_llm = call_eval_llm,
        temperature = temperature,
        eval_mode = eval_mode,
        theory_version = theory_version,
        image_style = image_style,
        result_dir = result_dir,
        file_name = file_name,
        system_prompt_name = system_prompt_name,
    )

def generate(
    gen_llm_name: str = "gemini-1.5-flash",
    dm_name: str = "stable-diffusion-3-medium-diffusers",
    dataset_name: str = "ours_gen_v1",
    prompt_name: str = "standard",
    api_key: str = "yz",
    n_per_topic: int = -1,
    gen_mode: str = "standard",
    n_selected_from: int = 2,
    eval_prompt_name: str = "theory",
    eval_llm_name: str = None,
    temperature: float = 0.5,
    eval_mode: Literal["description", "meme"] = "description",
    theory_version: str = 'v1',
    image_style: str = "cartoon",
    height: int = 480,
    width: int = 480,
    num_inference_steps: int = 28,
    guidance_scale: float = 7.0,
    negative_prompt: str = "",
    seed: int = 1234,
    n_words_in_filename: int = 5,
    max_new_tokens: int = 200,
    n_memes_per_content: int = 1,
    overwrite: bool = False,
    system_prompt_name: str = 'strict_scorer',
    data_mode: Literal["both", "train", "test"] = "both",
):
    if dataset_name not in support_gen_datasets:
        raise ValueError(f"Dataset {dataset_name} not supported!")
    if data_mode in ["train", "test"]:
        if not support_gen_datasets[dataset_name]["train_test_split"]:
            raise ValueError(f"Dataset {dataset_name} does not support train/test split!")
    if n_per_topic > 0 and not support_gen_datasets[dataset_name]["category"]:
        raise ValueError(f"Dataset {dataset_name} is not a category dataset, so n_per_topic must be -1!")
    

    dataset = load_dataset(dataset_name)
    if data_mode in ["train", "test"]: dataset = dataset[data_mode]

    call_gen_llm = load_model(f"{gen_llm_name}/pretrained", api_key)
    call_dm = load_model(f"{dm_name}", api_key)
    if gen_mode == "selective":
        call_eval_llm = load_model(f"{eval_llm_name}/pretrained", api_key)
    else:
        call_eval_llm = None
    result_dir = f"{root_dir}/results/generation/{dataset_name}/{gen_llm_name}/{dm_name}/{prompt_name}/{gen_mode}"


    contents, file_names = [], []
    if support_gen_datasets[dataset_name]["category"]:
        for topic in dataset:
            if n_per_topic == -1:
                iterations = range(len(dataset[topic]))
            else:
                iterations = range(min(n_per_topic, len(dataset[topic])))

            for i in iterations:
                content = dataset[topic][i]
                contents.append(content)
                file_names.append(f"{topic}_{i+1}")
    else:
        for i in dataset.index:
            content = dataset.loc[i]
            contents.append(content)
            file_names.append(f"{i+1}")

    generate_func = generate_meme_content if support_gen_datasets[dataset_name]["mode"] == "content" else generate_meme_topic

    for content, file_name in zip(contents, file_names):

            if os.path.exists(f"{result_dir}/output/{file_name}.json") and os.path.exists(f"{result_dir}/meme/{file_name}.png") and not overwrite:
                print(f"Meme {file_name} already exists, skipping...")
                continue

            generate_func(
                call_gen_llm,
                content,
                description_only = False,
                call_dm = call_dm,
                gen_llm_name = gen_llm_name,
                dm_name = dm_name,
                prompt_name = prompt_name,
                height = height,
                width = width,
                num_inference_steps = num_inference_steps,
                guidance_scale = guidance_scale,
                negative_prompt = negative_prompt,
                seed = seed,
                n_words_in_filename = n_words_in_filename,
                max_new_tokens = max_new_tokens,
                n_memes_per_content = n_memes_per_content,
                gen_mode = gen_mode,
                n_selected_from = n_selected_from,
                eval_prompt_name = eval_prompt_name,
                eval_llm_name = eval_llm_name,
                call_eval_llm = call_eval_llm,
                temperature = temperature,
                eval_mode = eval_mode,
                theory_version = theory_version,
                image_style = image_style,    
                result_dir = result_dir,
                file_name = file_name,
                system_prompt_name = system_prompt_name,
            )

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    llm_names = []
    for llm in support_llms:
        llm_names.extend(support_llms[llm])

    dm_names = []
    for dm in support_diffusers:
        dm_names.extend(support_diffusers[dm])

    parser.add_argument('--gen_llm_name', type=str, default='gpt-4o', choices=llm_names)
    parser.add_argument('--dm_name', type=str, default='stable-diffusion-3-medium-diffusers', choices=dm_names)
    parser.add_argument('--dataset_name', type=str, default='ours_gen_v1', choices=support_gen_datasets.keys())
    parser.add_argument('--prompt_name', type=str, default='standard', choices=prompt_processor_default['generation'].keys())
    parser.add_argument('--api_key', type=str, default='yz')
    parser.add_argument('--n_per_topic', type=int, default=-1, help = "Number of social contents to consider per topic")
    parser.add_argument('--gen_mode', type=str, default='standard', choices=['standard', 'selective'])
    parser.add_argument('--n_selected_from', type=int, default=2, help = "For each content, generate n_selected_from meme generations and select the best one")
    parser.add_argument('--eval_prompt_name', type=str, default='theory', choices=['theory', 'standard'])
    parser.add_argument('--eval_llm_name', type=str, default='gemini-1.5-flash', choices=llm_names)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--eval_mode', type=str, default='description', choices=['description', 'meme'])
    parser.add_argument('--theory_version', type=str, default='v4', choices=['v1', 'v2', 'v3', 'v4'])
    parser.add_argument('--image_style', type=str, default='realistic')
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--width', type=int, default=480)
    parser.add_argument('--num_inference_steps', type=int, default=28)
    parser.add_argument('--guidance_scale', type=float, default=7.0)
    parser.add_argument('--negative_prompt', type=str, default='')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--n_words_in_filename', type=int, default=5)
    parser.add_argument('--max_new_tokens', type=int, default=200)
    parser.add_argument('--n_memes_per_content', type=int, default=1)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--system_prompt_name', type=str, default='strict_scorer', choices=list(system_prompts_default.keys()))
    parser.add_argument('--data_mode', type=str, default='both', choices=['both', 'train', 'test'])
    args = parser.parse_args()

    print(__file__)
    print_configs(args)

    if args.wandb:
        wandb.init(
            project = WANDB_INFO_GEN['project'],
            entity = WANDB_INFO_GEN['entity'],
            config = vars(args),
        )
        wandb_step_log = {}

    generate(
        gen_llm_name = args.gen_llm_name,
        dm_name = args.dm_name,
        dataset_name = args.dataset_name,
        prompt_name = args.prompt_name,
        api_key = args.api_key,
        n_per_topic = args.n_per_topic,
        gen_mode = args.gen_mode,
        n_selected_from = args.n_selected_from,
        eval_prompt_name = args.eval_prompt_name,
        eval_llm_name = args.eval_llm_name,
        temperature = args.temperature,
        eval_mode = args.eval_mode,
        theory_version = args.theory_version,
        image_style = args.image_style,
        height = args.height,
        width = args.width,
        num_inference_steps = args.num_inference_steps,
        guidance_scale = args.guidance_scale,
        negative_prompt = args.negative_prompt,
        seed = args.seed,
        n_words_in_filename = args.n_words_in_filename,
        max_new_tokens = args.max_new_tokens,
        n_memes_per_content = args.n_memes_per_content,
        overwrite = args.overwrite,
        system_prompt_name = args.system_prompt_name,
        data_mode = args.data_mode,
    )
