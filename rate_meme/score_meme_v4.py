import pdb
import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from helper import save_json, read_json
from rate_meme.utils import get_score_v4

def score_meme_based_on_theory_v4(
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

    prompt = """
    Please objectively answer the following questions about the meme and provide an option from 0 to 3 for each.
    * Option Scale:
        - 0: Not at all
        - 1: A bit
        - 2: Moderately
        - 3: Very much
    * JSON Output Format:
        ```json
        { 
            "a": { "comment": "<your comment here>", "option": <0-3> }, 
            "b": { "comment": "<your comment here>", "option": <0-3> }, 
            "c": { "comment": "<your comment here>", "option": <0-3> }, 
            "d": { "comment": "<your comment here>", "option": <0-3> } 
        }
        ```
    * Evaluation Questions:
        a. Is there a clear and explicit setup that creates an expectation which is humorously contradicted by a surprising twist? Note: Only rate positively if the contradiction or twist is unexpected and adds humor. Default to 0 unless the setup and contradiction are explicitly evident.
        b. Does the meme explicitly include a surprising twist that adds cleverness or insight? Note: The twist should offer something beyond a simple answer—it must be creative or add a clever layer to the humor. Default to 0 unless the twist is genuinely clever.
        c. Does the meme contain something sensitive, offensive, or norm-breaking? Note: Evaluate if there are elements that could be seen as inappropriate or offensive in context. Score 0 if there is no norm-breaking or sensitive content.
        d. If the meme contains sensitive or offensive content, to what extent can the violation be interpreted as playful or non-threatening? Note: If there is no violation, assign an option of 0.
    * Example Scoring for Calibration:
        - Meme 1 (Morpheus Meme): "What if I told you that every 60 seconds in Africa... a minute passes":
            ```json
            { 
                "a": { "comment": "The meme sets up an expectation by using the familiar 'What if I told you' phrase. The humor lies in the expectation of something profound being humorously contradicted by a trivial fact.", "option": 1 }, 
                "b": { "comment": "The twist is surprising because it contrasts the serious tone with an obvious and mundane fact, adding a layer of clever humor.", "option": 1 }, 
                "c": { "comment": "There is no sensitive or potentially offensive content in the meme. It is benign and norm-abiding.", "option": 0 }, 
                "d": { "comment": "No violation of norms or negative content is present, so this question is not applicable.", "option": 0 } 
            }
            ```
        - Meme 2 (Rage Comic Meme): "When can we hang out?" - "IDK":
            ```json
            { 
                "a": { "comment": "There is a setup, but the answer 'IDK' is a typical response with no real contradiction or twist. The answer is mundane and expected.", "option": 0 }, 
                "b": { "comment": "There is no surprising twist or clever insight here. The answer is straightforward and lacks an element of humor or subversion.", "option": 0 }, 
                "c": { "comment": "The meme does not contain any offensive or norm-breaking content.", "option": 0 }, 
                "d": { "comment": "Since there is no offensive content, this question is not applicable.", "option": 0 } 
            }
            ```
    """

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
