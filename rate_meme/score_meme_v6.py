import pdb
import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from helper import save_json, read_json
from rate_meme.utils import get_score_v4


def score_v6_json_format(x = {
    "Q1_reasoning": "<your comment here>",
    "Q1_option":"<0-5>",
    "Q2_reasoning": "<your comment here>",
    "Q2_option": "<0-5>",
    "Q3_reasoning": "<your comment here>",
    "Q3_option": "<0-5>",
    "Q4_reasoning": "<your comment here>",
    "Q4_option": "<0-5>",
}):
    return f"""
    ```json
    {{ 
        "a": {{ "comment": "{x["Q1_reasoning"]}", "option": {x["Q1_option"]} }}, 
        "b": {{ "comment": "{x["Q2_reasoning"]}", "option": {x["Q2_option"]} }}, 
        "c": {{ "comment": "{x["Q3_reasoning"]}", "option": {x["Q3_option"]} }}, 
        "d": {{ "comment": "{x["Q4_reasoning"]}", "option": {x["Q4_option"]} }} 
    }}
    ```
    """

def prompt_score_v6(x = {
    "Q1_reasoning": "<your comment here>",
    "Q1_option":"<0-5>",
    "Q2_reasoning": "<your comment here>",
    "Q2_option": "<0-5>",
    "Q3_reasoning": "<your comment here>",
    "Q3_option": "<0-5>",
    "Q4_reasoning": "<your comment here>",
    "Q4_option": "<0-5>",
}):
    return f"""
    Please objectively answer the following questions about the meme and provide an option from 0 to 5 for each.
    * Option Scale:
        - 0: Not Applicable
        - 1: Strongly Disagree
        - 2: Disagree
        - 3: Neutral
        - 4: Agree
        - 5: Strongly Agree
    * JSON Output Format:
        {score_v6_json_format(x)}
    * Evaluation Questions:
        a. Does the meme present a clear setup that contradicts expectations?
        b. Does the contradiction in the image add cleverness or insight?
        c. Does the image contain violating elements?
        d. If any violating elements are present, are they benign (meaning playful or non-threatening)?
    """

def score_meme_based_on_theory_v6(
    meme_path,
    call_model,
    result_dir = None,
    max_new_tokens=1,
    example = False,
    description = '',
    context = '',
    overwrite = False,
    system_prompt_name = 'default',
):
    if example:
        raise ValueError("Example is not supported for score meme algorithm v4")

    if result_dir:
        img_name = meme_path.split("/")[-1].split(".")[0]
        example_flag = "example" if example else "plain"
        result_file = f'{result_dir}/scores/{example_flag}/{img_name}_v6.json'
        if os.path.exists(result_file) and not overwrite:
            return read_json(result_file)

    prompt = prompt_score_v6()

    outputs = get_score_v4(
        prompt = prompt,
        meme_path = meme_path,
        call_model = call_model,
        max_new_tokens = max_new_tokens,
        description = description,
        context = context,
        system_prompt_name = system_prompt_name,
    )

    scores = {}
    for key in outputs:
        scores[key] = outputs[key]["option"]

    result_dict = {
        'scores': scores,
        'outputs': outputs,
    }
    if result_dir:
        save_json(result_dict, result_file)
    return result_dict
