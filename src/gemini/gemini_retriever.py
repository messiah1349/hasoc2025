from dataclasses import dataclass
from typing import Literal, Optional
import pandas as pd
from src.common.constants import OUTPUT_PATH, CHROMA_STORAGE_PATH, GEMINI_EMBEDDINGS_MODEL, GEMINI_API_KEY
import logging
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_community.vectorstores.chroma import Chroma
from langchain_community.document_loaders import DataFrameLoader
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.utils.utils import convert_to_secret_str

logger = logging.getLogger(__name__)
content_typing = Literal['image', 'text', 'ocr']
dataset_type_typing = Literal['train', 'test', 'val']

languages = ['Bangla', 'Bodo', 'Gujarati', 'Hindi']


@dataclass
class RetrieverOuptut:
    id: str
    extracted_texts: list[Document]
    extracted_images: list[Document]


class GeminiRetriever:
    def __init__(self) -> None:
        self.document_embedder = GoogleGenerativeAIEmbeddings(
            model=GEMINI_EMBEDDINGS_MODEL, 
            google_api_key=convert_to_secret_str(GEMINI_API_KEY),
            task_type='RETRIEVAL_DOCUMENT',
        )
        self.query_embedder = GoogleGenerativeAIEmbeddings(
            model=GEMINI_EMBEDDINGS_MODEL, 
            google_api_key=convert_to_secret_str(GEMINI_API_KEY),
            task_type='RETRIEVAL_QUERY',
        )
        self.client = Chroma(persist_directory=str(CHROMA_STORAGE_PATH), embedding_function=self.document_embedder)

    @staticmethod
    def get_collection_id(
        language: str, 
        content_type: content_typing,
        dataset_type: dataset_type_typing,
    ) -> str:
        return f"{language}_{content_type}_{dataset_type}".lower()  

    def get_collection(self, collection_name: str) -> Chroma|None:

        try:
            collection = self.client._client.get_collection(name=collection_name)
            if collection.count() > 0:
                return Chroma(client=self.client._client, collection_name=collection_name, embedding_function=self.document_embedder)
        except Exception as e:
            logging.error(f"collection {collection_name} does not exists; exception: {e}")
            return None
            
    def _precalculate_embeddings_from_documents(self, collection_name: str, documents: list[Document]) -> Chroma:
        try:
            self.client._client.delete_collection(collection_name)
            logger.warning(f'collection {collection_name} was dropped')
        except Exception as e:
            pass

        logging.warning(f'start of embeddings calculation for {collection_name=}')
        vectorstore = Chroma.from_documents(
            documents,
            embedding=self.document_embedder,
            collection_name=collection_name,
            persist_directory=str(CHROMA_STORAGE_PATH),
        )
        logging.warning(f'embeddings calculation for {collection_name=} has finished')
        return vectorstore

    @staticmethod
    def _get_page_content_column(content_type: content_typing) -> str:
        if content_type == 'text':
            page_content_column = 'translated_text'
        elif content_type == 'image':
            page_content_column = 'image_description'
        elif content_type == 'ocr':
            page_content_column = 'OCR'
        return page_content_column
        

    def _prepare_documents(self, df: pd.DataFrame, content_type: content_typing) -> list[Document]:
        page_content_column = self._get_page_content_column(content_type)
        df[page_content_column] = df[page_content_column].fillna('not extracted')
        loader = DataFrameLoader(df, page_content_column=page_content_column)
        documents = loader.load()
        return documents

    def precalculate_train_embeddings(self, df: pd.DataFrame, language: str, 
                            content_types: list[str]=['text', 'image']) -> None:
        for content_type in content_types:
            collection_name = self.get_collection_id(language, content_type, 'train')
            documents = self._prepare_documents(df, content_type)
            _ = self._precalculate_embeddings_from_documents(collection_name, documents)
        return None

    def _calculate_documents_from_df(self, df: pd.DataFrame, content_type: content_typing) -> list[list[float]]:
        page_content_column = self._get_page_content_column(content_type)
        embeddings = self.query_embedder.embed_documents(df[page_content_column].fillna('not extracted').tolist())
        return embeddings
       
    def extract_similar_samples_from_train(self, test_df: pd.DataFrame, 
            language: str, k: int=5, content_types: list[str] = ['text', 'image']) -> list[RetrieverOuptut]:

        pre_data = {}

        for content_type in content_types:
            collection_name = self.get_collection_id(language, content_type, 'train')
            train_vectorstore = self.get_collection(collection_name)

            test_embeddings = self._calculate_documents_from_df(test_df, content_type)

            pre_data[content_type] = {'train_vectorstore': train_vectorstore, 'test_embeddings': test_embeddings}

        output_rows = []

        if not 'ocr' in content_types:
            for id, text_test_embedding, image_test_embedding in zip(test_df['Ids'], pre_data['text']['test_embeddings'], pre_data['image']['test_embeddings']): 
                retrieved_documents_text = pre_data['text']['train_vectorstore'].similarity_search_by_vector(
                    embedding=text_test_embedding,
                    k=k,
                )
                retrieved_documents_image = pre_data['image']['train_vectorstore'].similarity_search_by_vector(
                    embedding=image_test_embedding,
                    k=k,
                )
                retriever_output = RetrieverOuptut(id=id, extracted_texts=retrieved_documents_text, extracted_images=retrieved_documents_image)
                output_rows.append(retriever_output)

        else:
            for id, oct_embeddings in zip(test_df['Ids'], pre_data['ocr']['test_embeddings']):
                retrieved_documents_text = pre_data['ocr']['train_vectorstore'].similarity_search_by_vector(
                    embedding=oct_embeddings,
                    k=k,
                )
                retriever_output = RetrieverOuptut(id=id, extracted_texts=retrieved_documents_text, extracted_images=[])
                output_rows.append(retriever_output)

        return output_rows
        
