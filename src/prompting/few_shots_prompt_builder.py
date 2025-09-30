from dataclasses import dataclass

import pandas as pd
from langchain_core.documents import Document

from src.common.constants import PROMPT_FEW_SHOTS_TEMPLATE, DESCRIPTION_TEMPLATE, EXAMPLE_TEMPLATE, DESCRIPTION_OCR_TEMPLATE
from src.gemini.gemini_retriever import RetrieverOuptut

@dataclass
class FewShotsPromptBuilder:

    @staticmethod
    def _get_example(document: Document, content_types: list[str]) -> str:

        if 'ocr' in content_types:
            ocr = document.page_content
            description = DESCRIPTION_OCR_TEMPLATE.format(ocr=ocr)
        else:
            image_description = document.metadata['image_description'] if 'image_description' in document.metadata else document.page_content
            text_description = document.metadata['translated_text'] if 'translated_text' in document.metadata else document.page_content
            description = DESCRIPTION_TEMPLATE.format(
                image_description=image_description,
                text_description=text_description,
            )
        example = EXAMPLE_TEMPLATE.format(
            description=description,
            ground_truth=document.metadata['ground_truth'],
        ).replace('\n', ' ')
        return example

    @staticmethod
    def _prepare_examples(retriever_ouptut: RetrieverOuptut, content_types: list[str]) -> str:
        if 'ocr' in content_types:
            total_extracted = retriever_ouptut.extracted_texts
        else: 
            total_extracted = retriever_ouptut.extracted_images + retriever_ouptut.extracted_texts
        examples = '\n'.join([FewShotsPromptBuilder._get_example(document, content_types) for document in total_extracted])
        return examples

    @staticmethod
    def prepare_single_prompt(df_row, retriever_ouptut: RetrieverOuptut, content_types: list[str]) -> str:
        examples = FewShotsPromptBuilder._prepare_examples(retriever_ouptut, content_types)
        if 'ocr' in content_types:
            curr_description = DESCRIPTION_OCR_TEMPLATE.format(
                ocr=df_row.OCR,
            )
        else:
            curr_description = DESCRIPTION_TEMPLATE.format(
                image_description=df_row.image_description,
                text_description=df_row.translated_text,
            )
        prompt = PROMPT_FEW_SHOTS_TEMPLATE.format(
            examples=examples,
            meme_description=curr_description,
        )

        return prompt

    @staticmethod
    def prepare_prompts_for_df(df: pd.DataFrame, retriever_ouptuts: list[RetrieverOuptut], 
            content_types: list[str] = ['image', 'text']) -> list[str]:
        prompts = []

        for df_row, retriever_ouptut in zip(df.itertuples(), retriever_ouptuts):
            prompt = FewShotsPromptBuilder.prepare_single_prompt(df_row, retriever_ouptut, content_types)
            prompts.append(prompt)

        return prompts

