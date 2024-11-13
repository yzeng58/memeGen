import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from helper import save_json, read_json
from rate_meme.utils import get_score

def score_meme_based_on_theory_v1(
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
        result_file = f'{result_dir}/scores/{example_flag}/{img_name}_v1.json'
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
    humor_questions['ipr2'] = {
        "question": "Does it enable resolution of the incongruity through reinterpretation (finding a 'cognitive rule' that makes the surprising ending fit)?",
        "rating": "Please assign a score between 0 and 9, where 0 means no new understanding and 9 means highly satisfying realization.",
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
    ### Secondary Factors ###
    #########################

    ### * Diminishment & Reframing ###

    humor_questions['dr1'] = {
        "question": "How effectively does this meme reduce the importance/seriousness of its subject?",
        "rating": "Please give a score between 0 and 9, where 0 means no reduction and 9 means highly effective diminishment.",
    }

    humor_questions['dr2'] = {
        "question": "How successfully does this meme transform something serious into something humorous?",
        "rating": "Please assign a score between 0 and 9, where 0 means no transformation and 9 means perfect transformation.",
    }

    ### * Elaboration Potential ###

    humor_questions['ep1'] = {
        "question": "Can this meme be interpreted in multiple valid ways?",
        "rating": "Please provide a score between 0 and 9, where 0 means single interpretation and 9 means multiple rich interpretations.",
    }
    
    humor_questions['ep2'] = {
        "question": "How well does this meme connect to other memes, cultural references, or shared experiences?",
        "rating": "Please provide a score between 0 and 9, where 0 means no connections and 9 means rich connections.",
    }
    
    humor_questions['ep3'] = {
        "question": "What is the potential for creative variations or responses to this meme?",
        "rating": "Please provide a score between 0 and 9, where 0 means no potential and 9 means high potential.",
    }

    #########################
    ### Supporting Factor ###
    #########################

    ### * Integration of Elements ###

    humor_questions['ie1'] = {
        "question": "How well do the visual and textual elements work together in this meme?",
        "rating": "Please provide a score between 0 and 9, where 0 means poor integration and 9 means perfect integration.",
    }

    humor_questions['ie2'] = {
        "question": "Does the combination of elements create meaning beyond their individual parts?",
        "rating": "Please provide a score between 0 and 9, where 0 means no enhanced meaning and 9 means significant enhanced meaning.",
    }
    
    
    scores, outputs = {}, {}

    # Primary factors
    outputs["ipr1"] = get_score(
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
    scores["ipr1"] = outputs["ipr1"]["score"]

    outputs["vbn1"] = get_score(
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

    if scores["ipr1"] >= 6: 
        outputs["ipr2"] = get_score(
            humor_questions["ipr2"],
            meme_path = meme_path,
            call_model = call_model,
            output_control = output_control,
            example = False,
            max_intermediate_tokens = max_intermediate_tokens,
            max_new_tokens = max_new_tokens,
            description = description,
            context = context,
        )
        scores["ipr2"] = outputs["ipr2"]["score"]
        score_ipr = scores["ipr1"] * (1 + scores["ipr2"] * .01) / (1 + 9*.01)
    else:
        score_ipr = scores["ipr1"] 

    score_vbn = scores["vbn1"]
    if scores["vbn1"] >= 6:
        outputs["vbn2"] = get_score(
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

    for q in ["dr1", "dr2", "ep1", "ep2", "ep3", "ie1", "ie2"]:
        outputs[q] = get_score(
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

    score_secondary = scores["dr1"] + scores["dr2"] + scores["ep1"] + scores["ep2"] + scores["ep3"]
    score_supporting = scores["ie1"] + scores["ie2"]
    score_final = score_primary * (1 + .02 * score_secondary) * (1 + .005 * score_supporting)

    result_dict = {
        'output': score_final,
        'scores': scores,
        'outputs': outputs,
    }   
    
    if result_dir:
        save_json(result_dict, result_file)
    return result_dict
