from PIL import Image, ImageDraw, ImageFont
import requests, functools, time, pdb, json, os, random, torch, transformers, textwrap
from io import BytesIO, StringIO
import numpy as np
import pandas as pd
from IPython.display import display

root_dir = os.path.dirname(os.path.abspath(__file__))

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

def combine_text_and_image(image_path, upper_text, lower_text):
    img = Image.open(image_path)
    if img.mode == 'RGBA': img = img.convert('RGB')
    width, height = img.size
    
    draw = ImageDraw.Draw(img)
    
    def calculate_font_size(text, width):
        base_font_size = int(width/12)
        text_length = len(text)
        return max(base_font_size - int(text_length / 10), 10)
    
    def draw_multiline_text(text, y_start, font):
        # Split text into lines
        lines = text.split('\n')
        y = y_start
        
        for line in lines:
            # Get text size for this line
            text_bbox = draw.textbbox((0, 0), line, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            
            # Center this specific line
            x = (width - text_width) / 2
            
            # Draw outline
            for offset in range(-2, 3):
                for offset2 in range(-2, 3):
                    draw.text((x + offset, y + offset2), line, font=font, fill='black')
            draw.text((x, y), line, font=font, fill='white')
            
            # Move to next line
            y += font.size + 5  # Add small padding between lines
    
    # Add upper text
    if upper_text and not pd.isna(upper_text):
        upper_text = textwrap.fill(upper_text.upper(), width=25)
        font_size = calculate_font_size(upper_text, width)
        font = ImageFont.truetype(f"{root_dir}/utils/fonts/impact.ttf", font_size)
        draw_multiline_text(upper_text, height * 0.05, font)
    
    # Add lower text
    if lower_text and not pd.isna(lower_text):
        lower_text = textwrap.fill(lower_text.upper(), width=25)
        font_size = calculate_font_size(lower_text, width)
        font = ImageFont.truetype(f"{root_dir}/utils/fonts/impact.ttf", font_size)
        
        # Calculate total height of lower text to position it properly
        lines = lower_text.split('\n')
        total_height = len(lines) * (font.size + 5) - 5  # Subtract the last padding
        y_start = height * 0.95 - total_height
        
        draw_multiline_text(lower_text, y_start, font)
    
    img.save(image_path)

