import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage, TextChunk, ImageURLChunk, SystemMessage, AssistantMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

from huggingface_hub import snapshot_download

from configs import system_prompts
from helper import read_json, set_seed
import base64

import pdb

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def load_pixtral(model_name):
    pixtral_path = f"{root_dir}/models/pixtral/{model_name}"
    if not os.path.exists(pixtral_path):
        os.makedirs(pixtral_path)
        print("=="*10)
        print(f"Downloading {model_name} to {pixtral_path}...")
        snapshot_download(
            repo_id=f"mistralai/{model_name}",
            allow_patterns=["params.json", "consolidated.safetensors", "tekken.json"],
            local_dir=pixtral_path,
        )
        print(f"Downloaded {model_name} to {pixtral_path}!")
        print("=="*10)

    print("=="*10)
    print(f"Loading {model_name} from {pixtral_path}...")
    print("=="*10)

    model = Transformer.from_folder(pixtral_path)
    tokenizer = MistralTokenizer.from_file(f"{pixtral_path}/tekken.json")

    return {
        'model': model,
        'tokenizer': tokenizer,
    }

def process_sample_feature(
    description,
    context,
    image_paths,
):
    content = []
    for i, image_path in enumerate(image_paths):
        idx_str = f" {i+1}" if len(image_paths) > 1 else ""
        if description:
            content.append(TextChunk(text=f"Meme{idx_str}: {read_json(image_path)['description']}"))
        elif context:
            content.append(TextChunk(text=f"Meme{idx_str}: {read_json(image_path['description_path'])['description']}"))
            content.append(ImageURLChunk(
                image_url=f"data:image/{image_path['image_path'].split('.')[-1].lower()};base64,{encode_image(image_path['image_path'])}"
            ))
        else:
            content.append(ImageURLChunk(
                image_url=f"data:image/{image_path.split('.')[-1].lower()};base64,{encode_image(image_path)}"
            ))

    return content

def call_pixtral(
    pixtral,
    prompt,
    image_paths: list[str] = [],
    history = None,
    save_history = False,
    system_prompt = 'evaluator',
    description = '',
    max_new_tokens = 500,
    seed = 42,
    temperature = 0.1,
    context = "",
    demonstrations = None,
    **kwargs,
):
    set_seed(seed)
    model, tokenizer = pixtral['model'], pixtral['tokenizer']

    if history:
        messages = history
    else:
        messages = [
            SystemMessage(
                content=[TextChunk(text=system_prompts['pixtral'][system_prompt])]
            )
        ]

    if demonstrations:
        messages.append(UserMessage(content = [TextChunk(text=prompt)]))
        for sample in demonstrations:
            contents = process_sample_feature(
                description=description,
                context=context,
                image_paths=sample['image_paths'],
            )
            messages.append(UserMessage(content=contents))  

            if not 'label' in sample:
                raise ValueError("Label is required for non-test samples!")
            messages.append(AssistantMessage(content=[TextChunk(text=sample['label'])]))

    contents = process_sample_feature(
        description=description,
        context=context,
        image_paths=image_paths,
    )
    if not demonstrations: contents.append(TextChunk(text=prompt))
    messages.append(UserMessage(content=contents))

    request = ChatCompletionRequest(messages=messages)
    encoded = tokenizer.encode_chat_completion(request)
    images = encoded.images
    tokens = encoded.tokens

    out_tokens, _ = generate(
        [tokens],
        model,
        images=[images],
        max_tokens=max_new_tokens,
        temperature=temperature,
        eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
    )
    result = tokenizer.decode(out_tokens[0])

    output_dict = {}
    output_dict['output'] = result

    if save_history:
        messages.append(AssistantMessage(content=result))
        output_dict['history'] = messages

    return output_dict
    
    