from PIL import Image, ImageDraw, ImageFont
import requests, functools, time, pdb, json, os, random, torch, transformers, textwrap
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


def combine_text_and_image(image_path, upper_text, lower_text):
    img = Image.open(image_path)
    if img.mode == 'RGBA': img = img.convert('RGB')
    width, height = img.size
    
    draw = ImageDraw.Draw(img)
    
    def calculate_font_size(text, width):
        base_font_size = int(width/12)
        text_length = len(text)
        return max(base_font_size - int(text_length / 10), 10)  # Ensure font size is at least 10
    
    # Add upper text
    if upper_text and not pd.isna(upper_text):
        # Wrap text to fit width
        upper_text = textwrap.fill(upper_text.upper(), width=20)
        
        # Calculate font size
        font_size = calculate_font_size(upper_text, width)
        
        # Load font
        try:
            font = ImageFont.truetype("impact.ttf", font_size)
        except:
            # Fallback to default font if Impact not available
            font = ImageFont.load_default(font_size)
        
        # Get text size
        text_bbox = draw.textbbox((0, 0), upper_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        
        # Calculate position (centered horizontally, near top vertically)
        x = (width - text_width) / 2
        y = height * 0.05
        
        # Draw text with black outline
        for offset in range(-2, 3):
            for offset2 in range(-2, 3):
                draw.text((x + offset, y + offset2), upper_text, font=font, fill='black')
        draw.text((x, y), upper_text, font=font, fill='white')
    
    # Add lower text
    if lower_text and not pd.isna(lower_text):
        # Wrap text to fit width
        lower_text = textwrap.fill(lower_text.upper(), width=20)
        
        # Calculate font size
        font_size = calculate_font_size(lower_text, width)
        
        # Load font
        try:
            font = ImageFont.truetype("impact.ttf", font_size)
        except:
            # Fallback to default font if Impact not available
            font = ImageFont.load_default(font_size)
        
        # Get text size
        text_bbox = draw.textbbox((0, 0), lower_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Calculate position (centered horizontally, near bottom vertically)
        x = (width - text_width) / 2
        y = height * 0.85 - text_height
        
        # Draw text with black outline
        for offset in range(-2, 3):
            for offset2 in range(-2, 3):
                draw.text((x + offset, y + offset2), lower_text, font=font, fill='black')
        draw.text((x, y), lower_text, font=font, fill='white')
    
    img.save(image_path)