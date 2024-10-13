import os
root_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = f'{root_dir}/resources/datasets'

support_models = {
    'gpt': [
        'gpt-4o-mini',
    ],
    'claude': [
        'claude-3-haiku-20240307',
        'claude-3-sonnet-20240229',
    ],
    'llama': [
        "Llama-3.2-11B-Vision",
        "Llama-3.2-90B-Vision",
        "Llama-3.1-405B",
        "Llama-3.2-11B-Vision-Instruct",
        "Llama-3.2-90B-Vision-Instruct",
    ],
}

support_datasets = [
    'memotion',
    'ours_v2',
]

prompt_processor = {
    'single': {
        'yn': {
            'prompt': "Is this meme funny? Please respond with a single letter, 'Y' for yes, 'N' for no.",
            'output_processor': lambda x: x[-1].lower() == 'y',
        }
    },
    'pairwise': {
        'fs': {
            'prompt': "Which meme is more funny? Please respond with a single letter, 'F' for the first meme, 'S' for the second meme.",
            'output_processor': lambda x: int(x[-1].lower() == 's'),
        }
    }
}

description_prompt = {
    'default': "Describe this meme in detail. Include information about the image content, text content, and any cultural references or context that might be relevant to understanding the humor."
}
