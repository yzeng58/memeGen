import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from helper import save_json, read_json
from rate_meme.utils import get_score_v3

def score_meme_based_on_theory_v3(
    meme_path,
    call_model,
    result_dir = None,
    max_new_tokens=1,
    example = False,
    description = '',
    context = '',
    overwrite = False,
):
    if example:
        raise ValueError("Example is not supported for score meme algorithm v3")

    if result_dir:
        img_name = meme_path.split("/")[-1].split(".")[0]
        example_flag = "example" if example else "plain"
        result_file = f'{result_dir}/scores/{example_flag}/{img_name}_v1.json'
        if os.path.exists(result_file) and not overwrite:
            return read_json(result_file)

    output_control = """
    Please analyze the given meme using the evaluation criteria below. Provide the output in the following JSON format:
    {
        "Expectation_Punchline": {
            "comment": "<your comment here>"
            "score": <0-9>,
        },
        "Incongruity_Resolution": {
            "comment": "<your comment here>",
            "score": <0-9>,
        },
        "Norm_Violation": {
            "comment": "<your comment here>",
            "score": <0-9>,
        },
        "Playfulness": {
            "comment": "<your comment here>",
            "score": <0-9>,
        },
        "Reduction_of_Seriousness": {
            "comment": "<your comment here>",
            "score": <0-9>,
        },
        "Transformation_to_Humor": {
            "comment": "<your comment here>",
            "score": <0-9>,
        },
        "Ambiguity": {
            "comment": "<your comment here>",
            "score": <0-9>,
        },
        "Cultural_Connection": {
            "comment": "<your comment here>",
            "score": <0-9>,
        },
        "Creative_Potential": {
            "comment": "<your comment here>",
            "score": <0-9>,
        },
        "Visual_Textual_Synergy": {
            "comment": "<your comment here>",
            "score": <0-9>,
        },
        "Combined_Meaning": {
            "comment": "<your comment here>",
            "score": <0-9>,
        }
    }
    """
    humor_questions = """
    a. Expectation & Punchline: Does the setup create an initial expectation, followed by a punchline that subverts these expectations?

    b. Incongruity Resolution: Does it provide a reinterpretation that resolves the surprising ending in a meaningful way?

    c. Norm Violation: Does the meme contain something wrong, unexpected, or norm-breaking?

    d. Playfulness: To what extent can the violation be interpreted as playful or non-threatening?

    e. Reduction of Seriousness: How effectively does the meme downplay the seriousness of its subject?

    f. Transformation to Humor: How well does the meme transform a serious topic into something humorous?

    g. Ambiguity: Can the meme be interpreted in multiple valid ways?

    h. Cultural Connection: How well does it connect to other memes, cultural references, or shared experiences?

    i. Creative Potential: What is the potential for creative variations or responses to this meme?

    j. Visual & Textual Synergy: How well do the visual and textual elements work together?

    k. Combined Meaning: Does the combination of elements create a meaning beyond their individual parts?
    """
    

    outputs = get_score_v3(
        output_control + humor_questions,
        meme_path = meme_path,
        call_model = call_model,
        max_new_tokens = max_new_tokens,
        description = description,
        context = context,
    )

    scores = {}
    for key in [
        "Expectation_Punchline", 
        "Incongruity_Resolution", 
        "Norm_Violation", 
        "Playfulness", 
        "Reduction_of_Seriousness", 
        "Transformation_to_Humor", 
        "Ambiguity", 
        "Cultural_Connection", 
        "Creative_Potential", 
        "Visual_Textual_Synergy", 
        "Combined_Meaning"
    ]:
        scores[key] = outputs[key]["score"]

    if scores["Expectation_Punchline"] >= 6: 
        score_ipr = scores["Expectation_Punchline"] * (1 + scores["Incongruity_Resolution"] * .01) / (1 + 9*.01)
    else:
        score_ipr = scores["Expectation_Punchline"] 

    score_vbn = scores["Norm_Violation"]
    if scores["Norm_Violation"] >= 6:
        score_vbn = 9 - abs(scores["Norm_Violation"] - scores["Playfulness"])
    else:
        score_vbn = scores["Norm_Violation"] 

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

    score_secondary = scores["Reduction_of_Seriousness"] + scores["Transformation_to_Humor"] + scores["Ambiguity"] + scores["Cultural_Connection"] + scores["Creative_Potential"]
    score_supporting = scores["Visual_Textual_Synergy"] + scores["Combined_Meaning"]
    score_final = score_primary * (1 + .02 * score_secondary) * (1 + .005 * score_supporting)

    result_dict = {
        'output': score_final,
        'scores': scores,
        'outputs': outputs,
    }   
    
    if result_dir:
        save_json(result_dict, result_file)
    return result_dict
