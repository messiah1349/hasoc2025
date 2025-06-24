from pathlib import Path
from dataclasses import dataclass
from PIL import Image
from PIL.Image import Image as PILImage
from pydantic import BaseModel
from src.common.constants import GEMINI_API_KEY, PROMPT
from typing import Iterable, Literal
import pandas as pd

from google import genai


class MemeLLMOutput(BaseModel):
    sentiment: Literal['positive', 'Neutral', 'Negative']
    sarcasm: bool
    vulgar: bool
    abuse: bool
    target: Literal['gender', 'religion', 'individual', 'political', 'national origin', 'social subgroups', 'other', 'non-targeted']
    description: str


image_path = Path('/Users/evgenii/progs/competitions/hasoc2025/data/train/Screenshot 2025-06-24 at 19.51.55.png')
image = Image.open(image_path)

client = genai.Client(api_key=GEMINI_API_KEY)


class GeminiCaller:
    def __init__(self, model: str="gemini-2.5-flash", prompt: str=PROMPT):
        self.model: str=model
        self.prompt = prompt
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def call(self, image: PILImage) -> MemeLLMOutput:
        response = self.client.models.generate_content(
            model=self.model,
            contents=[self.prompt, image],
            config={
                "response_mime_type": "application/json",
                "response_schema": MemeLLMOutput,
            },    
        )
        return response.parsed

    def call_several_images(self, images: list[PILImage], ids: Iterable[str]) -> pd.DataFrame:
        responses = []
        for image, id in zip(images, ids):
            response = self.call(image).model_dump()
            response['id'] = id
            responses.append(response)
        response_df = pd.DataFrame(responses)
        return response_df
