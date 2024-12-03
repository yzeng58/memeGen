import pdb
import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from helper import save_json, read_json

def score_meme_based_on_theory_v5(
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
        result_file = f'{result_dir}/scores/{example_flag}/{img_name}_v4.json'
        if os.path.exists(result_file) and not overwrite:
            return read_json(result_file)

    questions = {
        "a": "Is there a clear and explicit setup that creates an expectation which is humorously contradicted by a surprising twist? Note: Only rate positively if the contradiction or twist is unexpected and adds humor. Default to 0 unless the setup and contradiction are explicitly evident.",
        "b": "Does the meme explicitly include a surprising twist that adds cleverness or insight? Note: The twist should offer something beyond a simple answer—it must be creative or add a clever layer to the humor. Default to 0 unless the twist is genuinely clever.",
        "c": "Does the meme contain something sensitive, offensive, or norm-breaking? Note: Evaluate if there are elements that could be seen as inappropriate or offensive in context. Score 0 if there is no norm-breaking or sensitive content.",
        "d": "If the meme contains sensitive or offensive content, to what extent can the violation be interpreted as playful or non-threatening? Note: If there is no violation, assign an option of 0."
    }
    
    output_control = """
    Please answer the question about the meme and provide an option from 0 to 3 for each.
    * Option Scale:
        - 0: Not at all
        - 1: A bit
        - 2: Moderately
        - 3: Very much
    Your answer should only contain the option number and nothing else.
    """

    outputs = {}
    for key in questions:
        prompt = f"{questions[key]}\n{output_control}"
        outputs[key] = call_model(
            questions[key] + prompt,
            [meme_path],
            max_new_tokens = max_new_tokens,
            description = description,
            context = context,
            system_prompt_name = system_prompt_name,
        )

    scores = {}
    for key in outputs:
        scores[key] = {"0": 0, "1": 1, "2": 2, "3": 3}.get(outputs[key]['output'].strip(), -1)

    result_dict = {
        'outputs': outputs,
        'scores': scores,
    }
    if result_dir:
        save_json(result_dict, result_file)
    return result_dict
