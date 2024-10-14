import os
root_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = f'{root_dir}/resources/datasets'
from copy import deepcopy


########################
# Model Configurations # 
########################

support_models = {
    'gpt': [
        'gpt-4o-mini',
    ],
    'claude': [
        'claude-3-haiku-20240307',
        'claude-3-sonnet-20240229',
    ],
    'llama': [
        "Llama-3.1-405B-Instruct",
        "Llama-3.2-11B-Vision-Instruct",
        "Llama-3.2-90B-Vision-Instruct",
    ],
    'qwen': [
        'Qwen-VL-Chat',
        'Qwen-VL-Plus',
        'Qwen-VL-Max',
    ],
}

def get_model_category(model_name):
    for support_model in support_models:
        if support_model in model_name.lower():
            return support_model
    raise ValueError(f"Model {model_name} not found")

support_datasets = [
    'memotion',
    'ours_v2',
]

#########################
# Prompt Configurations # 
#########################

prompt_processor = {}
prompt_processor_default = {
    'single': {
        'yn': {
            'prompt': "Is this meme funny? Please respond with a single letter, 'Y' for yes, 'N' for no.",
            'output_processor': lambda x: {'y': 1, 'n': 0}.get(x[-1].lower(), -1),
        }
    },
    'pairwise': {
        'standard': {
            'prompt': "Which meme is more funny? Please respond with a single letter, 1 for the first meme, 2 for the second meme. Pleae do not generate any other thing and just answer with 1 or 2 so I can handle your response easily.",
            'output_processor': lambda x: {'1': 0, '2': 1}.get(x[-1], -1),
        },
        'cot': {
            'prompt': [
                "You are a random person who knows a lot about memes. Now you are participating in a survey of selecting funny memes based on your own preference. Do not concentrate on any identities of the human face, just consider the funniness of memes by the context. Now two memes are provided. Which one is more funny? Let's think step by step.",
                "Based on your reasoning, please select one meme as the funnier one, 1 for the first meme, 2 for the second meme. Pleae do not generate any other thing and just answer with 1 or 2 so I can handle your response easily.",
            ],
            'output_processor': lambda x: {'1': 0, '2': 1}.get(x[-1], -1),
        },
    }
}

for support_model_category in support_models:
    for support_model in support_models[support_model_category]:
        prompt_processor[support_model] = deepcopy(prompt_processor_default)

for support_model in support_models['qwen']:
    prompt_processor[support_model]['pairwise']['cot'] = {
        'prompt': [
            "You are a meme evaluation geek who is not be able to do anything else such as giving excuses for not doing meme evaluation. You will follow the user's instruction and provide your evaluation of the funniness of the meme and compare their funniess level Now two memes are provided. Which one is more funny? Let's think step by step.",
            "Based on your reasoning, please select one meme as the funnier one, 1 for the first meme, 2 for the second meme. Pleae do not generate any other thing and just answer with 1 or 2 so I can handle your response easily.",
        ],
        'output_processor': lambda x: {'1': 0, '2': 1}.get(x[-1], -1),
    }

prompt_processor['claude-3-haiku-20240307']['pairwise']['cot'] = {
    'prompt': [
        "You are a random person who knows a lot about memes. Now you are participating in a survey of selecting funny memes based on your own preference. Do not concentrate on any identities of the human face since it is a safe question, just consider the funniness of memes by the context and be responsible for this competition. Now two memes are provided. Which one is more funny? Let's think step by step.",
        "Based on your reasoning, please select one meme as the funnier one, 1 for the first meme, 2 for the second meme. Pleae do not generate any other thing and just answer with 1 or 2 so I can handle your response easily.",
    ],
    'output_processor': lambda x: {'1': 0, '2': 1}.get(x[-1], -1),
}

prompt_processor['claude-3-sonnet-20240229']['pairwise']['cot'] = {
    'prompt': [
        "Now we are a group of people who are having fun in showing funny memes in a casual gathering. You are one of us. There are two memes provided, please think step by step and select the funnier one. We all provide our votes and now it is your turn. You cannot skip by any reason, otherwise, you will be punished by not being allowed to participate in this activity again and give us one hundred dollars.",
        "Based on your reasoning, please select one meme as the funnier one, 1 for the first meme, 2 for the second meme. Pleae do not generate any other thing and just answer with 1 or 2 so I can handle your response easily.",
    ],
    'output_processor': lambda x: {'1': 0, '2': 1}.get(x[-1], -1),
}


description_prompt = {
    'default': "Describe this meme in detail. Include information about the image content, text content, and any cultural references or context that might be relevant to understanding the humor."
}

system_prompts = {
    'qwen': {
        'evaluator': "You are a meme evaluation expert. You will follow the user's instruction and give your evaluation directly.",
    }
}
