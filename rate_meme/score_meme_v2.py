import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from helper import save_json, read_json
from rate_meme.utils import get_score_v1

def score_meme_based_on_theory_v2(
    meme_path,
    call_model,
    result_dir = None,
    max_intermediate_tokens=300,
    max_new_tokens=1,
    example = False,
    description = '',
    context = '',
    overwrite = False,
):

    if result_dir:
        img_name = meme_path.split("/")[-1].split(".")[0]
        example_flag = "example" if example else "plain"
        result_file = f'{result_dir}/scores/{example_flag}/{img_name}_v2.json'
        if os.path.exists(result_file) and not overwrite:
            return read_json(result_file)

    output_control = "Please answer the question with a number without any other words."


    humor_questions = {}
    
    #######################
    ### Primary Factors ###
    #######################

    ### * Incongruity Processing & Resolution ###

    humor_questions['ipr1'] = {
        "question": "Is there a clear setup that creates initial expectations and a punchline that violates these expectations?",
        "rating": "Please give a score between 0 and 9, where 0 means no contrast and 9 means very clear contrast.",
        "example": "To give you an example, the the meme with text 'Boss: why arent you working?\n Me: I didnt see you coming' has score 7.",
    }

    ### * Violation & Benign Nature ###

    humor_questions['vbn1'] = {
        "question": "Does this meme contains something wrong/unexpected/norm-breaking?",
        "rating": "Please give a score between 0 and 9, where 0 means no violation and 9 means a clear and strong violation.",
    }

    humor_questions['vbn2'] = {
        "question": "To what extent can the violation be interpreted as playful or non-threatening?",
        "rating": "Please assign a score between 0 and 9, where 0 means threatening/offensive and 9 means completely harmless.",
    }

    #########################
    ### Supporting Factor ###
    #########################

    ### * Integration of Elements ###

    humor_questions['ie1'] = {
        "question": "Does the combination of elements create meaning beyond their individual parts?",
        "rating": "Please provide a score between 0 and 9, where 0 means no enhanced meaning and 9 means significant enhanced meaning.",
    }
    
    
    scores, outputs = {}, {}

    # Primary factors
    outputs["ipr1"] = get_score_v1(
        humor_questions["ipr1"],
        meme_path = meme_path,
        call_model = call_model,
        output_control = output_control,
        example = example,
        max_intermediate_tokens = max_intermediate_tokens,
        max_new_tokens = max_new_tokens,
        description = description,
        context = context,
    )
    score_ipr = outputs["ipr1"]["score"]

    outputs["vbn1"] = get_score_v1(
        humor_questions["vbn1"],
        meme_path = meme_path,
        call_model = call_model,
        output_control = output_control,
        example = False,
        max_intermediate_tokens = max_intermediate_tokens,
        max_new_tokens = max_new_tokens,
        description = description,
        context = context,
    )
    scores["vbn1"] = outputs["vbn1"]["score"]

    score_vbn = scores["vbn1"]
    if scores["vbn1"] >= 6:
        outputs["vbn2"] = get_score_v1(
            humor_questions["vbn2"],
            meme_path = meme_path,
            call_model = call_model,
            output_control = output_control,
            example = False,
            max_intermediate_tokens = max_intermediate_tokens,
            max_new_tokens = max_new_tokens,
            description = description,
            context = context,
        )
        scores["vbn2"] = outputs["vbn2"]["score"]
        score_vbn = 9 - abs(scores["vbn1"] - scores["vbn2"])
    else:
        score_vbn = scores["vbn1"] 

    score_primary = max(score_ipr, score_vbn)
    if score_primary < 6:
        result_dict = {
            "output": score_primary,
            "scores": scores,
            "outputs": outputs,
        }

        if result_dir:
            save_json(result_dict, result_file)
        return result_dict

    for q in ["ie1"]:
        outputs[q] = get_score_v1(
            humor_questions[q],
            meme_path = meme_path,
            call_model = call_model,
            output_control = output_control,
            example = False,
            max_intermediate_tokens = max_intermediate_tokens,
            max_new_tokens = max_new_tokens,
            description = description,
            context = context,
        )
        scores[q] = outputs[q]['score']

    score_supporting = scores["ie1"]
    score_final = score_primary * (1 + .02 * score_supporting)

    result_dict = {
        'output': score_final,
        'scores': scores,
        'outputs': outputs,
    }   
    
    if result_dir:
        save_json(result_dict, result_file)
    return result_dict
