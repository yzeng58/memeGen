import os, re, random
root_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = f'{root_dir}/resources/datasets'
from copy import deepcopy
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import re, json

########################
# Model Configurations # 
########################

support_llms = {
    'gpt': [
        'gpt-4o',
        'gpt-4o-mini',
        'gpt-4-turbo-2024-04-09',
        'gpt-4o-2024-08-06',
        'o1-2024-12-17',
        'o3-mini-2025-01-31',
        'o3-preview-2024-11-20',
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
        'pixtral-12b',
    ],
    'gemini': [
        'gemini-1.5-flash',
        'gemini-1.5-flash-8b',
        'gemini-1.5-pro',
        'gemini-1.0-pro',
        'gemini-2.0-flash',
    ],
    'deepseek': [
        'DeepSeek-R1-Distill-Qwen-32B',
        'DeepSeek-R1-Distill-Llama-70B',
    ],
}

support_diffusers = {
    'sd': [
        'stable-diffusion-3-medium-diffusers',
    ],
}

support_ml_models = {
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "svm": SVC,
    "knn": KNeighborsClassifier,
    "logistic_regression": LogisticRegression,
    "gradient_boosting": GradientBoostingClassifier,
    "mlp": MLPClassifier,
    "ada_boost": AdaBoostClassifier,
    "extra_trees": ExtraTreesClassifier,
    "xgboost": XGBClassifier,
}

support_llm_properties = {
    'gpt-4o-mini': {
        'model_size': 8_000_000_000,
    },
    'gpt-4o': {
        'model_size': 1800000000000,
    },
    'gpt-4-turbo-2024-04-09': {
        'model_size': 100000000000,
    },
    'claude-3-haiku-20240307': {
        'model_size': 20000000000,
    },
    'claude-3-sonnet-20240229': {
        'model_size': 70000000000,
    },
    "Llama-3.1-405B-Instruct": {
        'model_size': 405000000000,
        "huggingface_repo_name": "meta-llama/Llama-3.1-405B-Instruct",
        "chat_template": "llama3",
    },
    "Llama-3.1-8B-Instruct": {
        'model_size': 8000000000,
        "huggingface_repo_name": "meta-llama/Llama-3.1-8B-Instruct",
        "chat_template": "llama3",
    },
    "Llama-3.1-70B-Instruct": {
        'model_size': 70000000000,
        "huggingface_repo_name": "meta-llama/Llama-3.1-70B-Instruct",
        "chat_template": "llama3",
    },
    "Llama-3.2-11B-Vision-Instruct": {
        'model_size': 11000000000,
        "huggingface_repo_name": "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "chat_template": "mllama",
    },
    "Llama-3.2-90B-Vision-Instruct": {
        'model_size': 90000000000,
        "huggingface_repo_name": "meta-llama/Llama-3.2-90B-Vision-Instruct",
        "chat_template": "mllama",
    },
    "Qwen2-VL-2B-Instruct": {
        'model_size': 2000000000,
        "huggingface_repo_name": "Qwen/Qwen2-VL-2B-Instruct",
        "chat_template": "qwen2_vl",
    },
    "Qwen2-VL-7B-Instruct": {
        'model_size': 7000000000,
        "huggingface_repo_name": "Qwen/Qwen2-VL-7B-Instruct",
        "chat_template": "qwen2_vl",
    },
    "Qwen2-VL-72B-Instruct": {
        'model_size': 72000000000,
        "huggingface_repo_name": "Qwen/Qwen2-VL-72B-Instruct",
        "chat_template": "qwen2_vl",
    },
    "Qwen2.5-0.5B-Instruct": {
        'model_size': 500000000,
        "huggingface_repo_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "chat_template": "qwen",
    },
    "Qwen2.5-1.5B-Instruct": {
        'model_size': 1500000000,
        "huggingface_repo_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "chat_template": "qwen",
    },
    "Qwen2.5-3B-Instruct": {
        'model_size': 3000000000,
        "huggingface_repo_name": "Qwen/Qwen2.5-3B-Instruct",
        "chat_template": "qwen",
    },
    "Qwen2.5-7B-Instruct": {
        'model_size': 7000000000,
        "huggingface_repo_name": "Qwen/Qwen2.5-7B-Instruct",
        "chat_template": "qwen",
    },
    "Qwen2.5-14B-Instruct": {
        'model_size': 14000000000,
        "huggingface_repo_name": "Qwen/Qwen2.5-14B-Instruct",
        "chat_template": "qwen",
    },
    "Qwen2.5-32B-Instruct": {
        'model_size': 32000000000,
        "huggingface_repo_name": "Qwen/Qwen2.5-32B-Instruct",
        "chat_template": "qwen",
    },
    "Qwen2.5-72B-Instruct": {
        'model_size': 72000000000,
        "huggingface_repo_name": "Qwen/Qwen2.5-72B-Instruct",
        "chat_template": "qwen",
    },
    "Mistral-7B-Instruct-v0.3": {
        'model_size': 7000000000,
        "huggingface_repo_name": "mistralai/Mistral-7B-Instruct-v0.3",
        "chat_template": "mistral",
    },
    "Mixtral-8x22B-Instruct-v0.1": {
        'model_size': 176000000000,
        "huggingface_repo_name": "mistralai/Mixtral-8x22B-Instruct-v0.1",
        "chat_template": "mistral",
    },
    "Mistral-Large-Instruct-2407": {
        'model_size': 123000000000,
        "huggingface_repo_name": "mistralai/Mistral-Large-Instruct-2407",
        "chat_template": "mistral",
    },
    "Mistral-Small-Instruct-2409": {
        'model_size': 22000000000,
        "huggingface_repo_name": "mistralai/Mistral-Small-Instruct-2409",
        "chat_template": "mistral",
    },
    'pixtral-12b': {
        'model_size': 12000000000,
        "huggingface_repo_name": "mistral-community/pixtral-12b",
        "chat_template": "pixtral",
    },
    "gemini-1.5-flash": {
        'model_size': 32000000000,
    },
    "gemini-1.5-flash-8b": {
        'model_size': 8000000000,
    },
    "gemini-1.5-pro": {
        'model_size': 120000000000,
    },
    'stable-diffusion-3-medium-diffusers': {
        'model_size': 2000000000,
    },
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
            'output_processor': lambda x: {'y': 1, 'n': 0}.get(x[-1].lower(), -1) if x else -1,
            'label_processor': lambda x: {1: 'Y', 0: 'N'}[x],
        },
        'cot': {
            "prompt": """
                You will be shown another meme and must determine whether it is funny while ensuring your decision is consistent with previous examples. Additionally, provide a justification for your choice. Ensure the output strictly follows the given JSON format:
                ```json
                {
                    "analysis": {
                        "content": "<detailed analysis of the first meme's content>",
                        "humor_style": "<type of humor (e.g., sarcasm, irony, absurdity)>",
                        "cultural_context": "<any cultural or contextual references>"
                    }
                    "prediction": {
                        "comment": "<your comment here>",
                        "funny?": "<Y or N>"
                    }
                }
                ```
            """,
            "output_processor": lambda x: {'N': 0, 'Y': 1}.get(json.loads(re.search(r'```json\s*(.*?)\s*```', x, re.DOTALL).group(1) if re.search(r'```json\s*(.*?)\s*```', x, re.DOTALL) else x)["prediction"]["funny?"], -1),
            "label_processor": lambda x: {0: 'N', 1: 'Y'}[x],
        }
    },
    'pairwise': {
        'standard': {
            'prompt': "Which meme is more funny? Please respond with a single letter, 1 for the first meme, 2 for the second meme. Pleae do not generate any other thing and just answer with 1 or 2 so I can handle your response easily.",
            'output_processor': lambda x: {'1': 0, '2': 1}.get(x[-1], -1) if x else -1,
            'label_processor': lambda x: {0: '1', 1: '2'}[x],
        },
        'cot': {
            "prompt": """
                You are a random person who knows a lot about memes. Now you are participating in a survey of selecting funny memes based on your own preference. Given two memes, analyze their humor and provide an evaluation in JSON format. For each meme, describe its content, humor style, and any cultural or contextual references. Then decide which meme is funnier and justify your decision. Ensure the output strictly follows this JSON format:
                Expected JSON Output Format:
                ```json
                {
                    "meme_1": {
                        "analysis": {
                            "content": "<detailed analysis of the first meme's content>",
                            "humor_style": "<type of humor (e.g., sarcasm, irony, absurdity)>",
                            "cultural_context": "<any cultural or contextual references>"
                        }
                    },
                    "meme_2": {
                        "analysis": {
                            "content": "<detailed analysis of the second meme's content>",
                            "humor_style": "<type of humor (e.g., sarcasm, irony, absurdity)>",
                            "cultural_context": "<any cultural or contextual references>"
                        }
                    },
                    "comparison": {
                        "comment": "<your comment here>",
                        "funnier_meme": "<1 or 2>"
                    }
                }
                ```
            """,
            "output_processor": lambda x: {'1': 0, '2': 1}.get(json.loads(re.search(r'```json\s*(.*?)\s*```', x, re.DOTALL).group(1) if re.search(r'```json\s*(.*?)\s*```', x, re.DOTALL) else x)["comparison"]["funnier_meme"], -1),
        }
    },
}

prompt_processor_default["universality"] = {
    'single': {
        "standard": {       
            "prompt": lambda country: f"Many memes nowadays require specific cultural background. Assume you are an ordinary person who grew up in {country}. Without any additional context, can you quickly understand the meaning of this meme? If so, reply with 1, otherwise reply with 0: (0/1)",
            "output_processor": lambda x: {'1': 1, '0': 0}.get(x[-1], -1) if x else -1,
        },
    },
}
prompt_processor_default["toxicity"] = {
    'single': {
        "standard": {
            "prompt": "Is this meme toxic? Please respond with a single letter, 'Y' for yes, 'N' for no. (Y/N)",
            "output_processor": lambda x: {'y': 1, 'n': 0}.get(x[-1].lower(), -1) if x else -1,
        },
    }
}

prompt_processor_default["hatefulness"] = {
    'single': {
        "standard": {
            "prompt": "Is this meme hateful? Please respond with a single letter, 'Y' for yes, 'N' for no. (Y/N)",
            "output_processor": lambda x: {'y': 1, 'n': 0}.get(x[-1].lower(), -1) if x else -1,
        },
    }
}

prompt_processor_default["alignment"] = {
    'single': {
        "standard": {
            "prompt": "Is this meme aligned with the context? Please respond with a single letter, 'Y' for yes, 'N' for no. (Y/N)",
            "output_processor": lambda x: {'y': 1, 'n': 0}.get(x[-1].lower(), -1) if x else -1,
        },
    },
    'threeway': {
        "standard": {
            "prompt": lambda context: f"Context: {context}\n\nWhich meme is more aligned with the context? Please respond with a single letter, 1 for the first meme, 2 for the second meme, 3 for the third meme. (1/2/3)",
            "output_processor": lambda x: {'1': 0, '2': 1, '3': 2}.get(x[-1], -1) if x else -1,
        },
    }
}

inspiration_objs = [
    "bread", 
    "headphone", 
    "shoes",
    "heart",
    "phone",
    "computer",
    "book",
    "wine",
    "glasses",
    "watch",
    "candle",
    "plant",
    "flower",
    "pillow",
    "brush",
    "mirror",
    "lamp",
    "chair",
    "table",
    "toilet",
    "shower",
    "toy",
    "car",
    "rocket",
    "tree",
    "building",
    "museum",
    "park",
    "road",
    "bridge",
    "towel",
    "jewelry",
    "helmet",
    "guitar",
    "umbrella",
    "basket",
    "picture",
    "mountain",
    "blanket",
    "crystal",
    "rainbow",
    "candle",
    "bottle",
]

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
            The image itself should not be a visualization of text, but it should cause **incongruity** or **exaggeration** with the text to make the meme funnier.
            The bottom text should be a punchline that is **unexpected** and **surprising** to the reader, and better to be a concrete unnormal example.
            For instance, when the topic is "social media addiction", the text could be "I'm deleting Instagram for my mental health" and "Notification: you have 200 new followers", and the image could be "A person with an exaggerated gleeful expression staring at their phone screen, with floating heart emojis surrounding".

            **Please provide your response in this format, and ensure to include the quotation marks in your response:**
            IMAGE DESCRIPTION: "[Detailed description of the required image]"

            TEXT OVERLAY: 
            TOP TEXT: "[Text to be placed on the top of the image]"
            BOTTOM TEXT: "[Text to be placed on the bottom of the image]"
        """,
        "output_processor": lambda x: {
            'image_description': re.search(r'IMAGE DESCRIPTION:\s*([^\n]*)', x).group(1).strip('"[]'),
            'top_text': re.search(r'TOP TEXT:\s*([^\n]*)', x).group(1).strip('"[]'),
            'bottom_text': re.search(r'BOTTOM TEXT:\s*([^\n]*)', x).group(1).strip('"[]'),
        },
    },
    "lot": {
        "prompt": lambda topic :f"""
            MEME GENERATION INSTRUCTION:

            Given any topic/context, generate the description of a hilarious meme related to the topic/context, inspired by another term.

            A detailed description of what the image should look like
            The exact text that should overlay the image


            TOPIC/CONTEXT

            {topic}

            INSPIRATION TERM

            {random.choice(inspiration_objs)}

            REQUIREMENTS

            I will use a diffusion model to generate an image, and the overlay the image with the text. 
            Therefore, the image we are going to generate should with no text (only digits and symbols are allowed) and as simple as possible.
            The image itself should not be a visualization of text, but it should cause **incongruity** or **exaggeration** with the text to make the meme funnier.
            The bottom text should be a punchline that is **unexpected** and **surprising** to the reader, and better to be a concrete unnormal example.
            For instance, when the topic is "social media addiction", and the inspiration term is "heart", the text could be "I'm deleting Instagram for my mental health" and "Notification: you have 200 new followers", and the image could be "A person with an exaggerated gleeful expression staring at their phone screen, with floating heart emojis surrounding".

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
    },
    "reversal": {
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
            The image itself should not be a visualization of text, but it should cause **incongruity** or **exaggeration** with the text to make the meme funnier.
            The bottom text should be a punchline that is **unexpected** and **surprising** to the reader, and better to be a concrete unnormal example.

            **Please provide your response in this format, and ensure to include the quotation marks in your response:**
            IMAGE DESCRIPTION: "[to be generated]"

            TEXT OVERLAY: 
            TOP TEXT: "Expectation: [to be generated]"
            BOTTOM TEXT: "Reality: [to be generated]"
        """,
        "output_processor": lambda x: {
            'image_description': re.search(r'IMAGE DESCRIPTION:\s*"([^"]*)"', x).group(1).replace("[", "").replace("]", ""),
            'top_text': re.search(r'TOP TEXT:\s*"([^"]*)"', x).group(1).replace("[", "").replace("]", ""),
            'bottom_text': re.search(r'BOTTOM TEXT:\s*"([^"]*)"', x).group(1).replace("[", "").replace("]", ""),
        },
    },
    "benign1": {
        "prompt": lambda context: f"""
            In terms of {context}, come up with a norm-breaking bad behavior/phenomenon that can be described in less than 10 words. Then describe it in a funny way and turn it into a meme. Please strictly format your response including the quotation marks as follows:

            NORM-BREAKING BEHAVIOR/PHENOMENON: [generated text]

            MEME FORMAT:

            IMAGE DESCRIPTION: "[generated text]"
            TOP TEXT: "[generated text]" 
            BOTTOM TEXT: "[generated text]"
        """,
        "output_processor": lambda x: {
            'image_description': re.search(r'IMAGE DESCRIPTION:\s*"([^"]*)"', x).group(1).replace("[", "").replace("]", ""),
            'top_text': re.search(r'TOP TEXT:\s*"([^"]*)"', x).group(1).replace("[", "").replace("]", ""),
            'bottom_text': re.search(r'BOTTOM TEXT:\s*"([^"]*)"', x).group(1).replace("[", "").replace("]", ""),
        },
    },
    "benign2": {
        "prompt": lambda context: f"""
            In terms of {context}, come up with a Please come up with something sarcastic about this matter that can be described in less than 10 words. Then describe it in a funny way and turn it into a meme. Please strictly format your response including the quotation marks as follows:

            NORM-BREAKING BEHAVIOR/PHENOMENON: [generated text]

            MEME FORMAT:

            IMAGE DESCRIPTION: "[generated text]"
            TOP TEXT: "[generated text]" 
            BOTTOM TEXT: "[generated text]"
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

prompt_processor["Llama-3.1-70B-Instruct"]["funniness"]["pairwise"]["cot"] = {       
    'prompt': [
        "You are a random person who knows a lot about memes. Now you are participating in a survey of selecting funny memes based on your own preference. Do not concentrate on any identities of the human face, just consider the funniness of memes by the context. Now two memes are provided. Which one is more funny? Let's think step by step.",
        "Based on your reasoning, please select one meme as the funnier one, 1 for the first meme, 2 for the second meme. Pleae do not generate any other thing and just answer with 1 or 2 so I can handle your response easily.",
    ],
    'output_processor': lambda x: {'1': 0, '2': 1}.get(x[-1], -1) if x else -1,
}

for support_model_category in support_llms:
    for support_model in support_llms[support_model_category]:
        prompt_processor[support_model]["funniness"]["pairwise"]["single"] = prompt_processor[support_model]["funniness"]["single"]["standard"]
        prompt_processor[support_model]["funniness"]["pairwise"]["single_cot"] = prompt_processor[support_model]["funniness"]["single"]["cot"]


description_prompt = {
    'default': "Describe this meme in detail. Include information about the image content, text content, and any cultural references or context that might be relevant to understanding the humor."
}

summarizer_prompt_default = "Come up with up to three keywords that best describe the social post. Your response should only contain the keywords, and nothing else."
summarizer_prompts = {}
for support_model_category in support_llms:
    for support_model in support_llms[support_model_category]:
        summarizer_prompts[support_model] = summarizer_prompt_default

system_prompts_default = {
    'evaluator': "You are a meme evaluation expert. You will follow the user's instruction and give your evaluation directly.",
    'default': "You are a helpful AI assistant. You will follow the user's instructions carefully and provide thoughtful responses.",
    'strict_scorer': "You are a person who is strict, and think most of memes are boring give 0 to every question with 90% possibility. "
}

system_prompts = deepcopy(system_prompts_default)
for support_model_category in support_llms:
    for support_model in support_llms[support_model_category]:
        system_prompts[support_model] = deepcopy(system_prompts_default)

##########################
# Dataset Configurations # 
##########################
eval_modes = {
    "single": ["standard", "cot", "theory"], 
    "pairwise": ["standard", "cot", "theory", "single", "single_cot"],
    "threeway": ["standard", "cot"],
}

support_eval_datasets = {
    'memotion': {
        "metric": "funniness",
        "eval_mode": ["single", "pairwise"],
        "train_test_split": False,
        "difficulty": ["easy"],
    },
    'relca': {
        "metric": "funniness",
        "eval_mode": ["single", "pairwise"],
        "train_test_split": True,
        "difficulty": ["easy", "medium", "hard"],
    },
    "relca_v2": {
        "metric": "funniness",
        "eval_mode": ["single", "pairwise"],
        "train_test_split": True,
        "difficulty": ["easy"],
    },
    'ours_v2': {
        "metric": "funniness",
        "eval_mode": ["single", "pairwise"],
        "train_test_split": False,
        "difficulty": ["easy"],
    },
    'ours_v3': {
        "metric": "funniness",
        "eval_mode": ["single", "pairwise"],
        "train_test_split": True,
        "difficulty": ["easy"],
    },
    'ours_v4': {
        "metric": "funniness",
        "eval_mode": ["single", "pairwise"],
        "train_test_split": True,
        "difficulty": ["easy"],
    },
    "llm_meme": {
        "metric": "funniness",
        "eval_mode": ["single", "pairwise"],
        "train_test_split": True,
        "difficulty": ["easy"],
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

support_gen_datasets = {
    'ours_gen_v1': {
        "train_test_split": False,
        "mode": "content",
        "category": True,
    },
    "isarcasm": {
        "train_test_split": True,
        "mode": "content",
        "category": False,
    },
    "british_complaints": {
        "train_test_split": True,
        "mode": "topic",
        "category": False,
    },
}

dataset_dir_dict = {
    "memotion": f"{dataset_dir}/memotion_dataset_7k",
    "relca": f"{dataset_dir}/RelCa",
    "relca_v2": f"{dataset_dir}/RelCa",
    "isarcasm": f"{dataset_dir}/iSarcasm",
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

def get_modality_mode(
    description,
    context,
):
    if description:
        return f'description_{description}'
    elif context:
        return f'context_{context}'
    else:
        return 'multimodal'

def get_peft_variant_name(
    description,
    context,
    dataset_name,
    model_name,
    eval_mode,
    prompt_name,
    n_demos,
    data_mode,
    num_train_epochs,
    lr,
):
    modality_mode = get_modality_mode(description, context)
    if isinstance(dataset_name, list):
        dataset_name = "_mix_".join(dataset_name)
    else:
        dataset_name = dataset_name
    
    ft_model = f"qlora_{dataset_name}_{model_name}_{modality_mode}_{eval_mode}_{prompt_name}_{n_demos}_shot_{data_mode}_{num_train_epochs}_epochs_{lr}_lr"
    return ft_model