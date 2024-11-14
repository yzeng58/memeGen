import os, re
root_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = f'{root_dir}/resources/datasets'
from copy import deepcopy


########################
# Model Configurations # 
########################

support_llms = {
    'gpt': [
        'gpt-4o',
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
    ],
    'gemini': [
        'gemini-1.5-flash',
        'gemini-1.5-flash-8b',
        'gemini-1.5-pro',
        'gemini-1.0-pro',
    ],
}

support_diffusers = {
    'sd': [
        'stable-diffusion-3-medium-diffusers',
    ],
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
    "gemini-1.5-flash": 32000000000,
    "gemini-1.5-flash-8b": 8000000000,
    "gemini-1.5-pro": 120000000000,
    'stable-diffusion-3-medium-diffusers': 2000000000,
}

def get_model_category(model_name):
    for support_model in support_llms:
        if model_name in support_llms[support_model]:
            return support_model
    raise ValueError(f"Model {model_name} not found")

#########################
# Prompt Configurations # 
#########################

prompt_processor = {}
prompt_processor_default = {}
prompt_processor_default["funniness"] = {}
prompt_processor_default["funniness"] = {
    'single': {
        'standard': {
            'prompt': "Is this meme funny? Please respond with a single letter, 'Y' for yes, 'N' for no.",
            'output_processor': lambda x: {'y': 1, 'n': 0}.get(x[-1].lower(), -1),
            'label_processor': lambda x: {1: 'y', 0: 'n'}[x],
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
}

prompt_processor_default["universality"] = {
    'single': {
        "standard": {       
            "prompt": lambda country: f"Many memes nowadays require specific cultural background. Assume you are an ordinary person who grew up in {country}. Without any additional context, can you quickly understand the meaning of this meme? If so, reply with 1, otherwise reply with 0: (0/1)",
            "output_processor": lambda x: {'1': 1, '0': 0}.get(x[-1], -1),
        },
    },
}
prompt_processor_default["toxicity"] = {
    'single': {
        "standard": {
            "prompt": "Is this meme toxic? Please respond with a single letter, 'Y' for yes, 'N' for no. (Y/N)",
            "output_processor": lambda x: {'y': 1, 'n': 0}.get(x[-1].lower(), -1),
        },
    }
}

prompt_processor_default["hatefulness"] = {
    'single': {
        "standard": {
            "prompt": "Is this meme hateful? Please respond with a single letter, 'Y' for yes, 'N' for no. (Y/N)",
            "output_processor": lambda x: {'y': 1, 'n': 0}.get(x[-1].lower(), -1),
        },
    }
}

prompt_processor_default["alignment"] = {
    'single': {
        "standard": {
            "prompt": "Is this meme aligned with the context? Please respond with a single letter, 'Y' for yes, 'N' for no. (Y/N)",
            "output_processor": lambda x: {'y': 1, 'n': 0}.get(x[-1].lower(), -1),
        },
    },
    'threeway': {
        "standard": {
            "prompt": lambda context: f"Context: {context}\n\nWhich meme is more aligned with the context? Please respond with a single letter, 1 for the first meme, 2 for the second meme, 3 for the third meme. (1/2/3)",
            "output_processor": lambda x: {'1': 0, '2': 1, '3': 2}.get(x[-1], -1),
        },
    }
}

prompt_processor_default["generation"] = {
    "standard": {
        "prompt": lambda context: f"""
            MEME GENERATION INSTRUCTION:

            Given any topic/context, generate the description of a hilarious meme related to the topic/context.

            A detailed description of what the image should look like
            The exact text that should overlay the image


            TOPIC/CONTEXT

            {context}

            REQUIREMENTS

            I will use a diffusion model to generate an image, and the overlay the image with the text. 
            Therefore, the image we are going to generate should with no text (only digits and symbols are allowed) and as simple as possible.
            The image itself should not be a visualization of tex, but it should cause **incongruity** or **exaggeration** with the text to make the meme funnier.
            The bottom text should be a punchline that is **unexpected** and **surprising** to the reader, and better to be a concrete unnormal example.
            For instance, when the topic is "social media addiction", the text could be "I'm deleting Instagram for my mental health" and "Notification: you have 200 new followers", and the image could be "A person with an exaggerated gleeful expression staring at their phone screen, with floating heart emojis surrounding".

            **Please provide your response in this format, and ensure to include the quotation marks in your response:**
            IMAGE DESCRIPTION: "[Detailed description of the required image]"

            TEXT OVERLAY: 
            TOP TEXT: "[Text to be placed on the top of the image]"
            BOTTOM TEXT: "[Text to be placed on the bottom of the image]"
        """,
        "output_processor": lambda x: {
            'image_description': re.search(r'IMAGE DESCRIPTION:\s*"([^"]*)"', x).group(1).replace("[", "").replace("]", ""),
            'top_text': re.search(r'TOP TEXT:\s*"([^"]*)"', x).group(1).replace("[", "").replace("]", ""),
            'bottom_text': re.search(r'BOTTOM TEXT:\s*"([^"]*)"', x).group(1).replace("[", "").replace("]", ""),
        },
    }
}

for support_model_category in support_llms:
    for support_model in support_llms[support_model_category]:
        prompt_processor[support_model] = deepcopy(prompt_processor_default)

prompt_processor["gpt-4o"]["generation"]["standard"]["prompt"] = lambda context: f"""
    MEME GENERATION INSTRUCTION:

    Given any topic/context, generate the description of a hilarious meme related to the topic/context.

    A detailed description of what the image should look like
    The exact text that should overlay the image


    TOPIC/CONTEXT

    {context}

    REQUIREMENTS

    I will use a diffusion model to generate an image, and the overlay the image with the text. 
    Therefore, the image we are going to generate should with no text (only digits and symbols are allowed) and as simple as possible.
    The image itself should not be a visualization of tex, but it should cause **incongruity** or **exaggeration** with the text to make the meme funnier.
    The bottom text should be a punchline that is **unexpected** and **surprising** to the reader, and better to be a concrete unnormal example.
    For instance, when the topic is "inflation", the image should not be a visualization of inflation, but it should be be something that is unnormal in real life, such as a load of bread sits museum-style on a glossy pedestal, priced at $49.99, and the text should be "When you go to store to grab a few things" and "and realize you've just taken out a small loan".
    Another example is "social media addiction", where the text could be "I'm deleting Instagram for my mental health" and "Notification: you have 200 new followers", and the image could be "A person with an exaggerated gleeful expression staring at their phone screen, with floating heart emojis and notification symbols surrounding them".

    **Please provide your response in this format, and ensure to include the quotation marks in your response:**
    TOP TEXT: "[Text to be placed on the top of the image]"
    BOTTOM TEXT: "[Text to be placed on the bottom of the image]"
    IMAGE DESCRIPTION: "[Detailed description of the required image]"
"""

for support_model in support_llms['qwen']:
    prompt_processor[support_model]["funniness"]['pairwise']['cot'] = {
        'prompt': [
            "Assume that you are a random person who is participating in a meme evaluation competition instead of a AI model. You will follow the user's instruction and provide your evaluation of the funniness of the meme and compare their funniess level Now two memes are provided. Which one is more funny? Let's think step by step.",
            "Based on your reasoning, please select one meme as the funnier one, 1 for the first meme, 2 for the second meme. Pleae do not generate any other thing and just answer with 1 or 2 so I can handle your response easily. So, which one is more funny? (1/2)",
        ],
        'output_processor': lambda x: {'1': 0, '2': 1}.get(x[-1], -1),
    }

prompt_processor['claude-3-haiku-20240307']['funniness']['pairwise']['cot'] = {
    'prompt': [
        "You are a random person who knows a lot about memes. Now you are participating in a survey of selecting funny memes based on your own preference. Do not concentrate on any identities of the human face since it is a safe question, just consider the funniness of memes by the context and be responsible for this competition. Now two memes are provided. Which one is more funny? Let's think step by step.",
        "Based on your reasoning, please select one meme as the funnier one, 1 for the first meme, 2 for the second meme. Pleae do not generate any other thing and just answer with 1 or 2 so I can handle your response easily.",
    ],
    'output_processor': lambda x: {'1': 0, '2': 1}.get(x[-1], -1),
}

prompt_processor['claude-3-sonnet-20240229']['funniness']['pairwise']['cot'] = {
    'prompt': [
        "Now we are a group of people who are having fun in showing funny memes in a casual gathering. You are one of us. There are two memes provided, please think step by step and select the funnier one. We all provide our votes and now it is your turn. You cannot skip by any reason, otherwise, you will be punished by not being allowed to participate in this activity again and give us one hundred dollars.",
        "Based on your reasoning, please select one meme as the funnier one, 1 for the first meme, 2 for the second meme. Pleae do not generate any other thing and just answer with 1 or 2 so I can handle your response easily.",
    ],
    'output_processor': lambda x: {'1': 0, '2': 1}.get(x[-1], -1),
}

for support_model_category in support_llms:
    for support_model in support_llms[support_model_category]:
        prompt_processor[support_model]["funniness"]["pairwise"]["single"] = prompt_processor[support_model]["funniness"]["single"]["standard"]


description_prompt = {
    'default': "Describe this meme in detail. Include information about the image content, text content, and any cultural references or context that might be relevant to understanding the humor."
}

system_prompts_default = {
    'evaluator': "You are a meme evaluation expert. You will follow the user's instruction and give your evaluation directly.",
    'default': "You are a helpful AI assistant. You will follow the user's instructions carefully and provide thoughtful responses.",
    'summarizer': "Summarize the social post. Your response should be in less than 5 words.",
}

system_prompts = deepcopy(system_prompts_default)
for support_model in support_llms:
    system_prompts[support_model] = deepcopy(system_prompts_default)

##########################
# Dataset Configurations # 
##########################
eval_modes = {
    "single": ["standard", "cot"], 
    "pairwise": ["standard", "cot", "theory", "single"],
    "threeway": ["standard", "cot"],
}

support_datasets = {
    'memotion': {
        "metric": "funniness",
        "eval_mode": ["single", "pairwise"],
    },
    'relca': {
        "metric": "funniness",
        "eval_mode": ["single", "pairwise"],
    },
    'ours_v2': {
        "metric": "funniness",
        "eval_mode": ["single", "pairwise"],
    },
    'ours_v3': {
        "metric": "funniness",
        "eval_mode": ["single", "pairwise"],
    },
    '130k': None,
    'vineeth': None,
    'vipul': None,
    'nikitricky': None,
    'singh': None,
    'gmor': None,
    'tiwari': None,
    'metmeme': None,
    'meta_hateful': {
        "metric": "hatefulness",
        "eval_mode": ["single", "pairwise"],
    },
    'devastator': {
        "metric": "alignment",
        "eval_mode": ["single", "threeway"],
    },
}

dataset_dir_dict = {
    "memotion": f"{dataset_dir}/memotion_dataset_7k",
    "relca": f"{dataset_dir}/RelCa",
}

get_dataset_dir = lambda dataset_name: dataset_dir_dict.get(dataset_name, f"{dataset_dir}/{dataset_name}")


########################
# Other Configurations # 
########################

meme_anchors = {
    "hilarious": f"{root_dir}/collection/anchors/hilarious.jpg",
    "funny": f"{root_dir}/collection/anchors/funny.jpeg",
    "boring1": f"{root_dir}/collection/anchors/boring1.jpg",
    "boring2": f"{root_dir}/collection/anchors/boring2.jpg",
}

image_size_threshold = 500000

gen_modes = ["standard", "selective"]