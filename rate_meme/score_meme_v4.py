from itertools import tee
import profile
import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from helper import save_json, read_json
from rate_meme.utils import get_score_v3

def score_meme_based_on_theory_v4(
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
        raise ValueError("Example is not supported for score meme algorithm v4")

    if result_dir:
        img_name = meme_path.split("/")[-1].split(".")[0]
        example_flag = "example" if example else "plain"
        result_file = f'{result_dir}/scores/{example_flag}/{img_name}_v4.json'
        if os.path.exists(result_file) and not overwrite:
            return read_json(result_file)

    output_control = """
    Please objectively answer the following questions about the meme and provide an option from 0 to 3 for each.
    * Option Scale:
        - 0: Not at all
        - 1: A bit
        - 2: Moderately
        - 3: Very much
    * Instructions:
        - Provide specific and detailed comments for each choice to justify your assessment.
        - Start by assuming a score of 0 unless there is strong, explicit evidence that justifies a higher score.
        - Assign low scores (0 or 1) by default unless the meme clearly demonstrates an effective setup, a twist, or an element that adds humor.
        - Be skeptical: Most memes should not score highly. Only assign a score of 2 or 3 if the meme is genuinely surprising, clever, and meets the criteria effectively.
        - Use the example scoring provided to calibrate your assessment of the meme's qualities. Remember: a high score should be rare and requires clear justification.
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
    * Guidelines for Avoiding High Scores:
Start with 0: Begin each evaluation by assuming the meme deserves a 0 and only increase the score if explicit evidence is found that meets the criteria.
Justification for High Scores: Before assigning a 2 or 3, ensure the meme has a significant, explicit, and clever twist that is genuinely funny and surprising. High scores are rare and must be clearly justified.
No Assumed Humor: If humor is not explicitly evident, default to 0 or 1. Simple answers or typical responses should not receive higher scores.
Usage Notes:
Use the above examples as references when evaluating new memes.
The comment field should provide a brief but specific rationale for the chosen option.
Options should reflect the actual qualities of the meme, avoiding score inflation and being grounded in clear evidence.
    """
    humor_questions = """
    a. Is there a setup with an expectation that is later humorously contradicted?

    b. Does the meme offer a surprising twist that is also clever or insightful?

    c. Is there an element in the meme that breaks common norms, expectations, or conventions?

    d. Can the norm violation in the meme be interpreted as harmless or playful rather than offensive or aggressive?

    e. Does the meme effectively use absurdity or exaggerated ridiculousness to create humor?

    f. How effectively do the visual and textual elements complement each other to deliver the intended humor?

    g. Can the meme be interpreted in more than one way, giving it a deeper or alternative layer of meaning?

    h. Does the meme reference shared cultural events, common experiences, or other memes, enhancing its humor through recognizability?

    i. Does the meme demonstrate originality in how it delivers its humor? Is there potential for interesting variations or responses?

    j. Does the combination of text, visuals, and context contribute to an overall meaning that is greater than the sum of its parts?
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
