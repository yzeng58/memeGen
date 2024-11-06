from PIL import Image
import requests, functools, time, pdb, json, os, random, torch, transformers
from io import BytesIO, StringIO
import numpy as np
import pandas as pd
from IPython.display import display

def save_json(data, path):
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)

    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
        
def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def read_jsonl(file_path):
    # Read jsonl file line by line
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                # Wrap the JSON string in StringIO to resolve deprecation warning
                data.append(pd.read_json(StringIO(line), typ='series'))
    return pd.DataFrame(data)


def retry_if_fail(func):
    @functools.wraps(func)
    def wrapper_retry(*args, **kwargs):
        retry = 0
        while retry <= 2:
            try:
                out = func(*args, **kwargs)
                break
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except pdb.bdb.BdbQuit:
                raise pdb.bdb.BdbQuit
            except Exception as e:
                retry += 1
                time.sleep(10)
                print(f"Exception occurred: {type(e).__name__}, {e.args}")
                print(f"Retry {retry} times...")

        if retry > 10:
            out = {'output': 'ERROR'}
            print('ERROR')
        
        return out
    return wrapper_retry

def get_image(image_path: str):
    if image_path.startswith('http://') or image_path.startswith('https://'):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")
    return image

def get_image_size(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
    return width*height

def display_image(image_path: str):
    image = get_image(image_path)
    display(image)

def print_configs(args):
    # print experiment configuration
    args_dict = vars(args)
    print("########"*3)
    print('## Experiment Setting:')
    print("########"*3)
    for key, value in args_dict.items():
        print(f"| {key}: {value}")
    print("########"*3)
    
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    transformers.set_seed(seed)

def score_meme(
    meme_path,
    call_model,
    max_intermediate_tokens=300,
    max_new_tokens=1,
):
    # # Cognitive Processing & Comprehension
    # cpc1 = "Is there a clear incongruity or surprise element?"
    # cpc2 = "Does it require/enable a reinterpretation?"
    # cpc3 = "Is it moderately difficult to understand (not too simple or too complex)?"

    # # Violation & Benign Nature
    # vbn1 = "Does it contain a violation (something wrong/unexpected/norm-breaking)?"
    # vbn2 = "Is this violation also perceived as harmless or non-threatening?"
    # vbn3 = "Does it maintain the 'sweet spot' between being threatening and benign?"

    # # Diminishment & Reframing
    # dr1 = "Does it make something seemingly important appear more trivial?"
    # dr2 = "Does it create a humorous reframing of the initial interpretation?"
    # dr3 = "Does it transform serious content into something playful?"

    # # Elaboration Potential
    # ep1 = "Can it generate multiple humor-relevant thoughts/interpretations?"
    # ep2 = "Does it connect to broader contexts or references?"
    # ep3 = "Can people build additional humor from the initial joke?"

    # # Integration of Elements
    # ie1 = "How well do visual and textual elements work together?"
    # ie2 = "Do multiple elements create additional layers of meaning?"
    # ie3 = "Does the combination enhance the humor?"

    # # (Ignore) Context & Relevance
    # cr1 = "Is it appropriate for its intended audience?"
    # cr2 = "Does it rely on shared cultural/social knowledge?"
    # cr3 = "Is it timely and relevant to current contexts?"

    output_control = "Please answer the question with a number without any other words."

    def process_score(score):
        try:
            return int(score)
        except:
            return -1

    
    def get_score(q):
        output_1 = call_model(
            q['question'],
            [meme_path],
            max_new_tokens=max_intermediate_tokens,
            save_history=True,
        )

        output_2 = call_model(
            q['rating'] + output_control,
            [],
            max_new_tokens = max_new_tokens,
            history = output_1['history'],
            save_history = True
        )

        output_dict = {
            'score': process_score(output_2['output']),
            'analysis': output_1['output'] + output_2['output'],
        }

        return output_dict


    humor_questions = {}
    
    #######################
    ### Primary Factors ###
    #######################

    ### * Cognitive Processing ###

    humor_questions['cp1'] = {
        "question": "Is there a clear contrast between initial and final interpretations?",
        "rating": "Please give a score between 0 and 10, where 0 means no contrast and 10 means very clear contrast.",
    }
    humor_questions['cp2'] = {
        "question": "Does the viewer arrive at a satisfying new understanding?",
        "rating": "Please assign a score between 0 and 10, where 0 means no new understanding and 10 means highly satisfying realization.",
    }
    humor_questions['cp3'] = {
        "question": "Is the meme at an appropriate difficulty level - neither too obvious nor too complex?",
        "rating": "Please provide a score between 0 and 10, where 0 means inappropriate difficulty and 10 means perfect difficulty level.",
    }

    ### * Violation & Benign Nature ###

    humor_questions['vbn1'] = {
        "question": "Does this meme contains something wrong/unexpected/norm-breaking?",
        "rating": "Please give a score between 0 and 10, where 0 means no violation and 10 means a clear and strong violation.",
        "example": "To give you an example, the the meme with text 'Boss: why arent you working?\n Me: I didnt see you coming' has score 7.",
    }

    humor_questions['vbn2'] = {
        "question": "To what extent can the violation be interpreted as playful or non-threatening?",
        "rating": "Please assign a score between 0 and 10, where 0 means threatening/offensive and 10 means completely harmless.",
    }

    humor_questions['vbn3'] = {
        "question": "How well does this meme balance being provocative yet acceptable?",
        "rating": "Please provide a score between 0 and 10, where 0 means poorly balanced and 10 means perfectly balanced.",
    }

    #########################
    ### Secondary Factors ###
    #########################

    ### * Diminishment & Reframing ###

    humor_questions['dr1'] = {
        "question": "How effectively does this meme reduce the importance/seriousness of its subject?",
        "rating": "Please give a score between 0 and 10, where 0 means no reduction and 10 means highly effective diminishment.",
    }

    humor_questions['dr2'] = {
        "question": "How successfully does this meme transform something serious into something humorous?",
        "rating": "Please assign a score between 0 and 10, where 0 means no transformation and 10 means perfect transformation.",
    }

    ### * Elaboration Potential ###

    humor_questions['ep1'] = {
        "question": "Can this meme be interpreted in multiple valid ways?",
        "rating": "Please provide a score between 0 and 10, where 0 means single interpretation and 10 means multiple rich interpretations.",
    }
    
    humor_questions['ep2'] = {
        "question": "How well does this meme connect to other memes, cultural references, or shared experiences?",
        "rating": "Please provide a score between 0 and 10, where 0 means no connections and 10 means rich connections.",
    }
    
    humor_questions['ep3'] = {
        "question": "What is the potential for creative variations or responses to this meme?",
        "rating": "Please provide a score between 0 and 10, where 0 means no potential and 10 means high potential.",
    }

    #########################
    ### Supporting Factor ###
    #########################

    ### * Integration of Elements ###

    humor_questions['ie1'] = {
        "question": "How well do the visual and textual elements work together in this meme?",
        "rating": "Please provide a score between 0 and 10, where 0 means poor integration and 10 means perfect integration.",
    }

    humor_questions['ie2'] = {
        "question": "Does the combination of elements create meaning beyond their individual parts?",
        "rating": "Please provide a score between 0 and 10, where 0 means no enhanced meaning and 10 means significant enhanced meaning.",
    }
    
    
    scores, outputs, score = {}, {}, 0

    # Primary factors
    outputs["cp1"] = get_score(humor_questions["cp1"])
    scores["cp1"] = outputs["cp1"]["score"]

    outputs["vbn1"] = get_score(humor_questions["vbn1"])
    scores["vbn1"] = outputs["vbn1"]["score"]

    score_cp = scores["cp1"]
    if scores["cp1"] >= 6: 
        for q in ["cp2", "cp3"]:
            outputs[q] = get_score(humor_questions[q])
            scores[q] = outputs[q]['score']
            score_cp += scores[q]

    score_vbn = scores["vbn1"]
    if scores["vbn1"] >= 6:
        for q in ["vbn2", "vbn3"]:
            outputs[q] = get_score(humor_questions[q])
            scores[q] = outputs[q]['score']
            score_vbn += scores[q]

    score_primary = max(score_cp, score_vbn)
    if score_primary < 18:
        return {
            "score": score_primary,
            "scores": scores,
            "outputs": outputs,
        }

    for q in ["dr1", "dr2", "ep1", "ep2", "ep3", "ie1", "ie2"]:
        outputs[q] = get_score(humor_questions[q])
        scores[q] = outputs[q]['score']

    score_secondary = scores["dr1"] + scores["dr2"] + scores["ep1"] + scores["ep2"] + scores["ep3"]
    score_supporting = scores["ie1"] + scores["ie2"]
    score_final = score_primary * (1 + .02 * score_secondary) * (1 + .005 * score_supporting)

    return {
        'score': score_final,
        'scores': scores,
        'outputs': outputs,
    }