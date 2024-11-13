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
    context: str = "Working hours are too long",
    prompt_name: str = "standard",
    max_new_tokens: int = 200,
    meme_path: str = None,
    temperature: float = 0.5,
    seed: int = 42,
):
    prompt = prompt_processor[gen_llm_name]["generation"][prompt_name]["prompt"](context)
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
        "context": context,
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
    context: str,
    prompt_name: str,
    max_new_tokens: int,
    short_context: str,
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
    meme_path = f"{result_dir}/meme/{short_context}_{time_string}.png"
    output_path = f"{result_dir}/output/{short_context}_{time_string}.json"
    os.makedirs(f"{result_dir}/meme", exist_ok=True)
    os.makedirs(f"{result_dir}/output", exist_ok=True)

    gen_llm_output_dict = generate_meme_llm(
        call_gen_llm = call_gen_llm,
        output_path = output_path,
        gen_llm_name = gen_llm_name,
        context = context,
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

def generate_meme_good(
    call_gen_llm,
    description_only: bool = False,
    call_dm = None,
    gen_llm_name: str = "gpt-4o-mini",
    dm_name: str = "stable-diffusion-3-medium-diffusers",
    context: str = "Working hours are too long",
    prompt_name: str = "standard",
    height: int = 300,
    width: int = 300,
    num_inference_steps: int = 28,
    guidance_scale: float = 7.0,
    negative_prompt: str = "",
    seed: int = 1234,
    n_words_in_filename: int = 5,
    max_new_tokens: int = 200,
    n_memes_per_context: int = 1,
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
        Generating a meme based on the following context:

        {context}
        """
    )

    set_seed(seed)
    context_no_punct = ''.join(c for c in context if c.isalnum() or c.isspace())
    short_context = "_".join(random.choices(context_no_punct.split(" "), k=n_words_in_filename))

    result_dir = f"{root_dir}/results/generation/{gen_llm_name}/{dm_name}/{prompt_name}"
    results = []

    for i in range(n_memes_per_context):
        seed_iter = random.randint(1, 10000)
        print(f"| Generating meme {i+1}")

        if gen_mode == "standard":
            output_dict = generate_meme_basic(
                result_dir = result_dir,
                call_gen_llm = call_gen_llm,
                gen_llm_name = gen_llm_name,
                context = context,
                prompt_name = prompt_name,
                max_new_tokens = max_new_tokens,
                short_context = short_context,
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
            output_path = f"{result_dir}/output/final_{short_context}_{time_string}.json"
            os.makedirs(f"{result_dir}/output", exist_ok=True)

            output_dict, best_idx = {}, 0
            for j in range(n_selected_from):
                seed_iter = random.randint(1, 10000)
                print(F"| -- For selective generation, generating meme {j+1}")
                gen_llm_output_dict = generate_meme_basic(
                    result_dir = result_dir,
                    call_gen_llm = call_gen_llm,
                    gen_llm_name = gen_llm_name,
                    context = context,
                    prompt_name = prompt_name,
                    max_new_tokens = max_new_tokens,
                    short_context = short_context,
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
                            context = "",
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
                            context = "",
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

def generate_batch_memes(
    gen_llm_name: str = "gpt-4o-mini",
    dm_name: str = "stable-diffusion-3-medium-diffusers",
    contexts: List[str] = ["Working hours are too long", "The boss is too mean"],
    gen_prompt_name: str = "standard",
    api_key: str = "yz",
    gen_mode: str = "standard",
    eval_llm_name: str = None,
    eval_prompt_name: str = "standard",
    n_memes_per_context: int = 1,
    n_selected_from: int = 3,
    height: int = 300,
    width: int = 300,
    num_inference_steps: int = 28,
    guidance_scale: float = 7.0,
    negative_prompt: str = "",
    n_words_in_filename: int = 5,
    max_new_tokens: int = 200,
    seed: int = 1234,
): 
    if gen_mode == "selective" and eval_llm_name is None:
        raise ValueError("eval_llm_name must be provided if gen_mode is selective!")

    call_gen_llm = load_model(gen_llm_name, api_key=api_key)
    call_dm = load_model(dm_name, api_key=api_key)

    for context in contexts:
        for _ in range(n_memes_per_context):
            generate_meme_good(
                call_gen_llm,
                gen_mode,
            )

    if gen_mode == "selective":
        call_eval_llm = load_model(eval_llm_name, api_key=api_key)
