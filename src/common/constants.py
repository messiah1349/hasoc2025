import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROMPT_PATH = BASE_DIR / 'configs' / 'prompt.yaml'
DATA_PATH = BASE_DIR / 'data'
TRAIN_PATH = DATA_PATH / 'train'
OUTPUT_PATH = DATA_PATH / 'output'

with open(PROMPT_PATH) as f:
    PROMPT = yaml.safe_load(f)['prompt']

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')



