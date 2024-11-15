import os, time, random
root_dir = os.path.dirname(__file__)
from configs import prompt_processor
from load_model import load_model
from helper import combine_text_and_image, set_seed, save_json
from rate_meme.rate_meme import score_meme_based_on_theory
from typing import List, Literal

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
): 
    time_string = time.strftime('%Y%m%d_%H%M%S')
    meme_path = f"{result_dir}/meme/{short_topic}_{time_string}.png"
    output_path = f"{result_dir}/output/{short_topic}_{time_string}.json"
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

    return gen_llm_output_dict

def summarize_topic(
    call_model,
    content, 
    max_new_tokens: int = 10,
):
    return call_model(
        prompt = content,
        image_paths = [],
        max_new_tokens = max_new_tokens,
        system_prompt = 'summarizer',
    )

def generate_meme_topic(
    call_gen_llm,
    description_only: bool = False,
    call_dm = None,
    gen_llm_name: str = "gpt-4o-mini",
    dm_name: str = "stable-diffusion-3-medium-diffusers",
    topic: str = "Working hours are too long",
    prompt_name: str = "standard",
    height: int = 300,
    width: int = 300,
    num_inference_steps: int = 28,
    guidance_scale: float = 7.0,
    negative_prompt: str = "",
    seed: int = 1234,
    n_words_in_filename: int = 5,
    max_new_tokens: int = 200,
    n_memes_per_topic: int = 1,
    gen_mode: str = "standard",
    n_selected_from: int = 2,
    eval_prompt_name: str = "theory",
    eval_llm_name: str = None,
    call_eval_llm = None,
    temperature: float = 0.5,
    eval_mode: Literal["description", "meme"] = "description",
    theory_version: str = 'v1',
    image_style: str = "cartoon",
): 
    if gen_mode == "selective":
        if eval_llm_name is None:
            raise ValueError("eval_llm_name must be provided if gen_mode is selective!")
        if call_eval_llm is None:
            raise ValueError("call_eval_llm must be provided if gen_mode is selective!")
        if n_selected_from <= 1:
            raise ValueError("n_selected_from must be greater than 1 if gen_mode is selective!")
        if not eval_prompt_name in ["theory"]:
            raise ValueError(f"eval_prompt_name {eval_prompt_name} not supported!")
        
    if not description_only and call_dm is None:
        raise ValueError("call_dm must be provided if description_only is False!")

    print(
        f"""
        Generating a meme based on the following topic:

        {topic}
        """
    )

    set_seed(seed)
    topic_no_punct = ''.join(c for c in topic if c.isalnum() or c.isspace())
    topic_words = topic_no_punct.split()
    n_words = min(n_words_in_filename, len(topic_words))
    selected_indices = sorted(random.sample(range(len(topic_words)), n_words))
    short_topic_words = [topic_words[i] for i in selected_indices]
    short_topic = "_".join(short_topic_words)

    result_dir = f"{root_dir}/results/generation/{gen_llm_name}/{dm_name}/{prompt_name}"
    results = []

    for i in range(n_memes_per_topic):
        seed_iter = random.randint(1, 10000)
        print(f"| Generating meme {i+1}")

        if gen_mode == "standard":
            output_dict = generate_meme_basic(
                result_dir = result_dir,
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
            time_string = time.strftime('%Y%m%d_%H%M%S')
            output_path = f"{result_dir}/output/final_{short_topic}_{time_string}.json"
            os.makedirs(f"{result_dir}/output", exist_ok=True)

            output_dict, best_idx = {}, 0
            for j in range(n_selected_from):
                seed_iter = random.randint(1, 10000)
                print(F"| -- For selective generation, generating meme {j+1}")
                gen_llm_output_dict = generate_meme_basic(
                    result_dir = result_dir,
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

                if eval_prompt_name == "theory":
                    if eval_mode == "description":
                        eval_llm_output_dict = score_meme_based_on_theory(
                            meme_path = gen_llm_output_dict['output_path'],
                            call_model = call_eval_llm,
                            result_dir = result_dir,
                            max_intermediate_tokens = 300,
                            max_new_tokens = 1,
                            example = True,
                            description = gen_llm_output_dict['description'],
                            topic = "",
                            theory_version = theory_version,
                        )
                    elif eval_mode == "meme":
                        eval_llm_output_dict = score_meme_based_on_theory(
                            meme_path = gen_llm_output_dict['meme_path'],
                            call_model = call_eval_llm,
                            result_dir = result_dir,
                            max_intermediate_tokens = 300,
                            max_new_tokens = 1,
                            example = True,
                            description = "",
                            topic = "",
                            theory_version = theory_version,
                        )

                output_dict[j+1] = {
                    "gen_llm_output": gen_llm_output_dict,
                    "eval_llm_output": eval_llm_output_dict,
                }

                if eval_llm_output_dict['output'] > output_dict[best_idx+1]['eval_llm_output']['output']:
                    best_idx = j
            
            output_dict["best_idx"] = best_idx+1
            save_json(output_dict, output_path)

            print(f"Selected meme stored in {output_dict[best_idx+1]['gen_llm_output']['meme_path']}")
        else:
            raise ValueError(f"gen_mode {gen_mode} not supported!")
        
        results.append(output_dict)
    return results

def generate_meme_content(
    call_gen_llm,
    description_only: bool = False,
    call_dm = None,
    gen_llm_name: str = "gpt-4o-mini",
    dm_name: str = "stable-diffusion-3-medium-diffusers",
    content: str = "Working hours are too long",
    prompt_name: str = "standard",
    height: int = 300,
    width: int = 300,
    num_inference_steps: int = 28,
    guidance_scale: float = 7.0,
    negative_prompt: str = "",
    seed: int = 1234,
    n_words_in_filename: int = 5,
    max_new_tokens: int = 200,
    n_memes_per_topic: int = 1,
    gen_mode: str = "standard",
    n_selected_from: int = 2,
    eval_prompt_name: str = "theory",
    eval_llm_name: str = None,
    call_eval_llm = None,
    temperature: float = 0.5,
    eval_mode: Literal["description", "meme"] = "description",
    theory_version: str = 'v1',
    image_style: str = "cartoon",    
):
    topic = summarize_topic(
        call_model = call_gen_llm,
        content = content,
        max_new_tokens = 20,
        model_name = gen_llm_name,
    )['output']

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
        n_memes_per_topic = n_memes_per_topic,
        gen_mode = gen_mode,
        n_selected_from = n_selected_from,
        eval_prompt_name = eval_prompt_name,
        eval_llm_name = eval_llm_name,
        call_eval_llm = call_eval_llm,
        temperature = temperature,
        eval_mode = eval_mode,
        theory_version = theory_version,
        image_style = image_style,
    )