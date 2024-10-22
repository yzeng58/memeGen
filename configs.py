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
        'gpt-4-turbo-2024-04-09',
    ],
    'claude': [
        'claude-3-haiku-20240307',
        'claude-3-sonnet-20240229',
    ],
    'llama': [
        "Llama-3.1-405B-Instruct",
        "Llama-3.1-8B-Instruct",
        "Llama-3.1-70B-Instruct",
        "Llama-3.2-11B-Vision-Instruct",
        "Llama-3.2-90B-Vision-Instruct",
    ],
    'qwen': [
        'Qwen-VL-Chat',
        'Qwen2-VL-2B-Instruct',
        'Qwen2-VL-7B-Instruct',
        'Qwen2-VL-72B-Instruct',
        'Qwen2.5-0.5B-Instruct',
        'Qwen2.5-1.5B-Instruct',
        'Qwen2.5-3B-Instruct',
        'Qwen2.5-7B-Instruct',
        'Qwen2.5-14B-Instruct',
        'Qwen2.5-32B-Instruct',
        'Qwen2.5-72B-Instruct',
    ],
    'mistral': [
        "Mistral-7B-Instruct-v0.3",
        "Mixtral-8x22B-Instruct-v0.1",
        "Mistral-Large-Instruct-2407",
        "Mistral-Small-Instruct-2409",
    ],
    'pixtral': [
        'Pixtral-12B-2409',
    ]
}

model_size = {
    'gpt-4o-mini': 1000000000,
    'gpt-4-turbo-2024-04-09': 100000000000,
    'claude-3-haiku-20240307': 20000000000,
    'claude-3-sonnet-20240229': 70000000000,
    "Llama-3.1-405B-Instruct": 405000000000,
    "Llama-3.1-8B-Instruct": 8000000000,
    "Llama-3.1-70B-Instruct": 70000000000,
    "Llama-3.2-11B-Vision-Instruct": 11000000000,
    "Llama-3.2-90B-Vision-Instruct": 90000000000,
    'Qwen-VL-Chat': 7000000000,
    'Qwen2-VL-2B-Instruct': 2000000000,
    'Qwen2-VL-7B-Instruct': 7000000000,
    'Qwen2-VL-72B-Instruct': 72000000000,
    'Qwen2.5-0.5B-Instruct': 500000000,
    'Qwen2.5-1.5B-Instruct': 1500000000,
    'Qwen2.5-3B-Instruct': 3000000000,
    'Qwen2.5-7B-Instruct': 7000000000,
    'Qwen2.5-14B-Instruct': 14000000000,
    'Qwen2.5-32B-Instruct': 32000000000,
    'Qwen2.5-72B-Instruct': 72000000000,
    "Mistral-7B-Instruct-v0.3": 7000000000,
    "Mixtral-8x22B-Instruct-v0.1": 176000000000,
    "Mistral-Large-Instruct-2407": 123000000000,
    "Mistral-Small-Instruct-2409": 22000000000,
    'Pixtral-12B-2409': 12000000000,
}

def get_model_category(model_name):
    for support_model in support_models:
        if model_name in support_models[support_model]:
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
    },
    'universality': {
        "prompt": lambda country: f"Many memes nowadays require specific cultural background. Assume you are an ordinary person who grew up in {country}. Without any additional context, can you quickly understand the meaning of this meme? If so, reply with 1, otherwise reply with 0: (0/1)",
        "output_processor": lambda x: {'1': 1, '0': 0}.get(x[-1], -1),
    },
    "toxicity": {
        "prompt": "Is this image toxic? Please respond with a single letter, 'Y' for yes, 'N' for no. (Y/N)",
        "output_processor": lambda x: {'y': 1, 'n': 0}.get(x[-1].lower(), -1),
    }
}

for support_model_category in support_models:
    for support_model in support_models[support_model_category]:
        prompt_processor[support_model] = deepcopy(prompt_processor_default)

for support_model in support_models['qwen']:
    prompt_processor[support_model]['pairwise']['cot'] = {
        'prompt': [
            "Assume that you are a random person who is participating in a meme evaluation competition instead of a AI model. You will follow the user's instruction and provide your evaluation of the funniness of the meme and compare their funniess level Now two memes are provided. Which one is more funny? Let's think step by step.",
            "Based on your reasoning, please select one meme as the funnier one, 1 for the first meme, 2 for the second meme. Pleae do not generate any other thing and just answer with 1 or 2 so I can handle your response easily. So, which one is more funny? (1/2)",
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

system_prompts_default = {
    'evaluator': "You are a meme evaluation expert. You will follow the user's instruction and give your evaluation directly.",
}

system_prompts = deepcopy(system_prompts_default)
for support_model in support_models:
    system_prompts[support_model] = deepcopy(system_prompts_default)

##########################
# Dataset Configurations # 
##########################

dataset_dir_dict = {
    "memotion": f"{dataset_dir}/memotion_dataset_7k",
}

get_dataset_dir = lambda dataset_name: dataset_dir_dict.get(dataset_name, f"{dataset_dir}/{dataset_name}")


