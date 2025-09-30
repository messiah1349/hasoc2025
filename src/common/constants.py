import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROMPT_PATH = BASE_DIR / 'configs' / 'prompt.yaml'
DATA_PATH = BASE_DIR / 'data'
TRAIN_PATH = DATA_PATH / 'train'
TEST_PATH = DATA_PATH / 'test'
OUTPUT_PATH = DATA_PATH / 'output'
MODELS_OUTPUT_PATH = OUTPUT_PATH / 'models'

BANGLA_TRAIN_IMAGES_PATH = TRAIN_PATH / 'Bangla_train_2025' / 'Bangla_train_images' 
BODO_TRAIN_IMAGES_PATH = TRAIN_PATH / 'Bodo_train_2025' / 'Bodo_train_images' 
GUJARATI_TRAIN_IMAGES_PATH = TRAIN_PATH / 'Gujarati_train_2025' / 'Gujarati_train_images' 
HINDI_TRAIN_IMAGES_PATH = TRAIN_PATH / 'Hindi_train_2025' / 'Hindi_train_images' 

BANGLA_TEST_IMAGES_PATH = TEST_PATH / 'Bangla_test_2025' / 'Bangla_test_images' 
BODO_TEST_IMAGES_PATH = TEST_PATH / 'Bodo_test_2025' / 'Bodo_test_images' 
GUJARATI_TEST_IMAGES_PATH = TEST_PATH / 'Gujarati_test_2025' / 'Gujarati_test_images' 
HINDI_TEST_IMAGES_PATH = TEST_PATH / 'Hindi_test_2025' / 'Hindi_test_images' 

with open(PROMPT_PATH) as f:
    prompt_dict = yaml.safe_load(f)

PROMPT = prompt_dict['prompt']
OCR_PROMPT = prompt_dict['ocr_prompt']
PROMPT_FEW_SHOTS_TEMPLATE = prompt_dict['prompt_few_shots']
DESCRIPTION_TEMPLATE = prompt_dict['description_template']
DESCRIPTION_OCR_TEMPLATE = prompt_dict['description_ocr_template']
EXAMPLE_TEMPLATE = prompt_dict['example_template']

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
GEMINI_MODEL = 'gemini-2.5-flash-lite-preview-06-17'
GEMINI_EMBEDDINGS_MODEL = "models/embedding-001"

CHROMA_STORAGE_PATH = BASE_DIR / 'storage' / 'chroma_embeddings'
