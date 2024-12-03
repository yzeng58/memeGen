import json, pdb, os

def process_score(score):
    try:
        return int(score)
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except:
        return -1


def get_score_v1(
    q, 
    meme_path,
    call_model,
    output_control = "Please answer the question with a number without any other words.",
    example = False,
    max_intermediate_tokens = 300,
    max_new_tokens = 1,
    description = '',
    context = '',
    system_prompt_name = 'default',
):
    output_1 = call_model(
        f"{q['question']} {q['rating']}" + (" " + q['example'] if example else ''),
        [meme_path],
        max_new_tokens=max_intermediate_tokens,
        save_history=True,
        description = description,
        context = context,
        system_prompt = system_prompt_name,
    )

    output_2 = call_model(
        q['rating'] + output_control,
        [],
        max_new_tokens = max_new_tokens,
        history = output_1['history'],
        save_history = True,
        description = description,
        context = context,
        system_prompt = system_prompt_name,
    )

    output_dict = {
        'score': process_score(output_2['output']),
        'analysis': output_1['output'] + output_2['output'],
    }

    return output_dict

def get_score_v3(
    prompt, 
    meme_path,
    call_model,
    max_new_tokens = 500,
    description = '',
    context = '',
    system_prompt_name = 'default',
):
    output = call_model(
        prompt, 
        [meme_path], 
        max_new_tokens=max_new_tokens, 
        save_history=True, 
        description = description, 
        context = context,
        system_prompt = system_prompt_name,
    )

    try:
        parsed_data = json.loads(output['output'].replace("```json", "").replace("```", "").strip())
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except:
        print(f"Error parsing the output of {os.path.basename(meme_path)}, using default values")
        parsed_data = {
            'Expectation_Punchline': {
                'comment': '',
                'score': -1,
            },
            'Incongruity_Resolution': {
                'comment': '',
                'score': -1,
            },
            'Norm_Violation': {
                'comment': '',
                'score': -1,
            },
            'Playfulness': {
                'comment': '',
                'score': -1,
            },
            'Reduction_of_Seriousness': {
                'comment': '',
                'score': -1,
            },
            'Transformation_to_Humor': {
                'comment': '',
                'score': -1,
            },
            'Ambiguity': {
                'comment': '',
                'score': -1,
            },
            'Cultural_Connection': {
                'comment': '',
                'score': -1,
            },
            'Creative_Potential': {
                'comment': '',
                'score': -1,
            },
            'Visual_Textual_Synergy': {
                'comment': '',
                'score': -1,
            },
            'Combined_Meaning': {
                'comment': '',
                'score': -1,
            },
        }

    return parsed_data


def get_score_v4(
    prompt, 
    meme_path,
    call_model,
    max_new_tokens = 500,
    description = '',
    context = '',
    system_prompt_name = 'default',
):
    output = call_model(
        prompt, 
        [meme_path], 
        max_new_tokens=max_new_tokens, 
        save_history=True, 
        description = description, 
        context = context,
        system_prompt = system_prompt_name,
    )
    pdb.set_trace()

    try:
        parsed_data = json.loads(output['output'].replace("```json", "").replace("```", "").strip())
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except:
        print(f"Error parsing the output of {os.path.basename(meme_path)}, using default values")
        parsed_data = {
            "a": { "comment": "", "option": -1 },
            "b": { "comment": "", "option": -1 },
            "c": { "comment": "", "option": -1 },
            "d": { "comment": "", "option": -1 },
        }

    return parsed_data
