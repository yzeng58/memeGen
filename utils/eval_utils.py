import os, sys
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

from rate_meme.rate_meme import score_meme_based_on_theory
import pdb

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
    system_prompt_name = "default",
):
    if prompt_name == "cot_old":
        output_1 = call_model(
            prompt[0], 
            image_paths, 
            max_new_tokens=max_intermediate_tokens,
            save_history=True,
            description=description,
            context=context,
            demonstrations = demonstrations,
            system_prompt = system_prompt_name,
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
            system_prompt = system_prompt_name,
        )
        output_dict = {
            'output': output_2['output'],
            'analysis': output_1['output'] + output_2['output'],
        }
    elif prompt_name in ["standard", "cot"]:
        new_tokens = max_new_tokens if prompt_name == "standard" else max_intermediate_tokens
        output_dict_all = call_model(
            prompt, 
            image_paths, 
            max_new_tokens=new_tokens,
            description=description,
            context=context,
            demonstrations = demonstrations,
            system_prompt = system_prompt_name,
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
            system_prompt_name = system_prompt_name,
        )
    else:
        raise ValueError(f"Prompt name {prompt_name} not supported")
    return output_dict
