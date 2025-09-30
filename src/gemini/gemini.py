from pathlib import Path
from dataclasses import dataclass
from PIL import Image
from PIL.Image import Image as PILImage
from pydantic import BaseModel
from src.common.constants import GEMINI_API_KEY, GEMINI_MODEL, PROMPT, OCR_PROMPT
from typing import Iterable, Literal, Any
import pandas as pd
from tqdm import tqdm
import logging

from google import genai
from google.genai.errors import ServerError


logger = logging.getLogger(__name__)

class MemeLLMOutput(BaseModel):
    sentiment: Literal['positive', 'neutral', 'negative']
    sarcasm: bool
    vulgar: bool
    abuse: bool
    target: Literal['gender', 'religion', 'individual', 'political', 'national origin', 'social subgroups', 'other', 'non-targeted']
    description: str


class TextAndImage(BaseModel):
    location: str
    image_description: str|None
    text: str|None
    translated_text: str|None


class GeminiCaller:
    def __init__(self, model: str=GEMINI_MODEL, prompt: str=PROMPT, return_type: Literal['meme_output', 'ocr', 'fs_call'] = 'meme_output'):
        self.model: str=model
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.return_type = return_type
        self.prompt = self._get_prompt()

    def _get_prompt(self) -> str:
        if self.return_type == 'meme_output':
            return PROMPT
        elif self.return_type == 'ocr':
            return OCR_PROMPT
        else:
            return ''
        
    def _gemini_config(self) -> dict[str, Any]:
        return {
            "response_mime_type": "application/json",
            "response_schema": list[TextAndImage] if self.return_type == 'ocr' else MemeLLMOutput,
        }

    def call(self, image: PILImage) -> MemeLLMOutput:
        response = self.client.models.generate_content(
            model=self.model,
            contents=[self.prompt, image],
            config=self._gemini_config(),    
        )
        return response.parsed

    def call_text(self, prompt: str) -> MemeLLMOutput:
        response = self.client.models.generate_content(
            model=self.model,
            contents=[prompt],
            config=self._gemini_config(),
        )
        return response.parsed

    def response_to_df(self, responses: list[dict[str, Any]]) -> pd.DataFrame:
        if self.return_type in ['meme_output', 'fs_call']:
            return  pd.DataFrame(responses)
        elif self.return_type == 'ocr':
            output_items = []
            for response in responses:
                if 'items' in response and 'id' in response:
                    items = response['items']
                    id = response['id']
                    for item in items:
                        item['id'] = id
                        output_items.append(item)
            return pd.DataFrame(output_items)

    def call_several_images(self, images: list[PILImage], ids: Iterable[str]) -> pd.DataFrame:
        responses = []
        for image, id in tqdm(zip(images, ids), total=len(ids)):
            try:
                response = self.call(image)
            except ServerError as e:
                logger.error(f"internal gemini server error appeared: {e}")
                response = None
            except Exception as e:
                logger.error(f"unusual error appeared: {e}")
                response = None
            # logging.warning(f'{response=}')
            try:
                if isinstance(response, list):
                    response = {'items': [item.model_dump() for item in response]}
                else:
                    response = response.model_dump()
                # logging.warning(f'{response=}')
            except Exception as e:
                response = {}
                logging.warning(f'error during response parse: {e}')
            response['id'] = id
            responses.append(response)
        response_df = self.response_to_df(responses)
        return response_df

    def call_fs_prompts(self, df: pd.DataFrame) -> pd.DataFrame:
        prompts = df['few_shot_prompt'] 
        ids = df['Ids']
        responses = []
        for prompt, id in tqdm(zip(prompts, ids), total=len(ids)):
            try:
                response = self.call_text(prompt)
            except ServerError as e:
                logger.error(f"internal gemini server error appeared: {e}")
                response = None
            except Exception as e:
                logger.error(f"unusual error appeared: {e}")
                response = None
            # logging.warning(f'{response=}')
            try:
                if isinstance(response, list):
                    response = {'items': [item.model_dump() for item in response]}
                else:
                    response = response.model_dump()
                # logging.warning(f'{response=}')
            except Exception as e:
                response = {}
                logging.warning(f'error during response parse: {e}')
            response['id'] = id
            responses.append(response)
        response_df = self.response_to_df(responses)
        return response_df
