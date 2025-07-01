import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROMPT_PATH = BASE_DIR / 'configs' / 'prompt.yaml'
DATA_PATH = BASE_DIR / 'data'
TRAIN_PATH = DATA_PATH / 'train'
OUTPUT_PATH = DATA_PATH / 'output'
BANGLA_TRAIN_IMAGES_PATH = TRAIN_PATH / 'Bangla_train_2025' / 'Bangla_train_images' 
BODO_TRAIN_IMAGES_PATH = TRAIN_PATH / 'Bodo_train_2025' / 'Bodo_train_images' 
GUJARATI_TRAIN_IMAGES_PATH = TRAIN_PATH / 'Gujarati_train_2025' / 'Gujarati_train_images' 
HINDI_TRAIN_IMAGES_PATH = TRAIN_PATH / 'Hindi_train_2025' / 'Hindi_train_images' 

with open(PROMPT_PATH) as f:
    prompt_dict = yaml.safe_load(f)

PROMPT = prompt_dict['prompt']
OCR_PROMPT = prompt_dict['ocr_prompt']

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
GEMINI_MODEL = 'gemini-2.5-flash-lite-preview-06-17'

