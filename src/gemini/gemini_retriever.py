import pandas as pd
from src.common.constants import OUTPUT_PATH
from langchain.embeddings import GeminiEmbedder
from langchain.vectorstores import Chroma

class LangchainContextRetriever:
    def __init__(self, languages, output_path):
        self.languages = languages
        self.output_path = output_path
        self.embedder = GeminiEmbedder()
        self.chroma_db = Chroma()

    def setup_chroma(self, train_df, test_df):
        for language in self.languages:
            for df, df_type in [(train_df, 'train'), (test_df, 'test')]:
                for text_type in ['image_description', 'translated_text']:
                    section_name = f"{language}_{df_type}_{text_type}"
                    texts = df[text_type].tolist()
                    embeddings = self.embedder.embed(texts)
                    self.chroma_db.add_section(section_name, embeddings)

    def retrieve_similar_samples(self, test_sample, language):
        results = {}
        for text_type in ['image_description', 'translated_text']:
            section_name = f"{language}_train_{text_type}"
            query_embedding = self.embedder.embed([test_sample[text_type]])[0]
            similar_samples = self.chroma_db.query(section_name, query_embedding, top_k=5)
            results[text_type] = similar_samples
        return results

    def prepare_few_shot_prompt(self, test_sample, language):
        similar_samples = self.retrieve_similar_samples(test_sample, language)
        few_shot_examples = []
        for text_type, samples in similar_samples.items():
            few_shot_examples.extend(samples)
        return few_shot_examples


languages = ['Bangla', 'Bodo', 'Gujarati', 'Hindi']

for language in languages:
    train_df = pd.read_csv(OUTPUT_PATH / f"{language.capitalize()}_train_ocr_concated.csv")
    test_df = pd.read_csv(OUTPUT_PATH / f"{language.capitalize()}_test_ocr_concated.csv")
