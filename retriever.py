import pandas as pd
import random
from src.common.constants import OUTPUT_PATH
from src.prompting.few_shots_prompt_builder import FewShotsPromptBuilder
from src.gemini.gemini_prepare_for_retriever import concat_ocr_files_by_language, concat_ocr_files_by_language_original_ocr
from src.gemini.gemini_retriever import GeminiRetriever, languages

def show_collection_5_samples(collection: str) -> None:
    all_ids = collection.get(include=[])["ids"]

    k = 5

    if len(all_ids) < k:
        raise ValueError("Not enough documents in the collection to get 5 random samples.")

    random_ids = random.sample(all_ids, k)

    random_documents_data = collection.get(
        ids=random_ids,
        include=["documents", "metadatas"]
    )

    print("Randomly selected documents:")
    for i in range(len(random_documents_data["ids"])):
        print(f"ID: {random_documents_data['ids'][i]}")
        print(f"Document: {random_documents_data['documents'][i]}")
        print(f"Metadata: {random_documents_data['metadatas'][i]}")
        print(f"Ground truth: {random_documents_data['metadatas'][i]['ground_truth']}")
        print("-" * 20)


# concat_ocr_files_by_language()

def precalculate_for_gemini_extracted_ocr():
    gemini_retriever = GeminiRetriever()

    for language in languages:
# for language in ['Bodo']:
        train_df = pd.read_csv(OUTPUT_PATH / f"{language.capitalize()}_train_ocr_concated.csv")
        val_df = pd.read_csv(OUTPUT_PATH / f"{language.capitalize()}_val_ocr_concated.csv")
        test_df = pd.read_csv(OUTPUT_PATH / f"{language.capitalize()}_test_ocr_concated.csv")

        gemini_retriever.precalculate_train_embeddings(train_df, language)

        # test_df = test_df.head().copy()

        test_retriever_outputs = gemini_retriever.extract_similar_samples_from_train(test_df, language)

        prompts = FewShotsPromptBuilder.prepare_prompts_for_df(test_df, test_retriever_outputs)
        test_df['few_shot_prompt'] = prompts
        test_df.to_csv(OUTPUT_PATH / f'{language}_test_fsprompts.csv', index=False)


        val_retriever_outputs = gemini_retriever.extract_similar_samples_from_train(val_df, language)

        # ex_images = test_retriever_outputs[1].extracted_texts
        # ex_image = ex_images[0]
        # ex_metadata = ex_image.metadata
        # print(ex_metadata)

        prompts = FewShotsPromptBuilder.prepare_prompts_for_df(val_df, val_retriever_outputs)
        val_df['few_shot_prompt'] = prompts
        val_df.to_csv(OUTPUT_PATH / f'{language}_val_fsprompts.csv', index=False)

def precalculate_for_original_ocr():

    gemini_retriever = GeminiRetriever()

    # for language in languages:
    for language in ['Bangla', 'Bodo']:
# for language in ['Bodo']:
        train_df = pd.read_csv(OUTPUT_PATH / f"{language.capitalize()}_train_original_ocr.csv")
        val_df = pd.read_csv(OUTPUT_PATH / f"{language.capitalize()}_val_original_ocr.csv")
        test_df = pd.read_csv(OUTPUT_PATH / f"{language.capitalize()}_test_original_ocr.csv")

        gemini_retriever.precalculate_train_embeddings(train_df, language, content_types=['ocr'])

        # test_df = test_df.head().copy()

        test_retriever_outputs = gemini_retriever.extract_similar_samples_from_train(test_df, language, k=10, content_types=['ocr'])

        prompts = FewShotsPromptBuilder.prepare_prompts_for_df(test_df, test_retriever_outputs, content_types=['ocr'])
        test_df['few_shot_prompt'] = prompts
        test_df.to_csv(OUTPUT_PATH / f'{language}_test_ocr_fsprompts.csv', index=False)


        val_retriever_outputs = gemini_retriever.extract_similar_samples_from_train(val_df, language, k=10, content_types=['ocr'])

        # ex_images = test_retriever_outputs[1].extracted_texts
        # ex_image = ex_images[0]
        # ex_metadata = ex_image.metadata
        # print(ex_metadata)

        prompts = FewShotsPromptBuilder.prepare_prompts_for_df(val_df, val_retriever_outputs, content_types=['ocr'])
        val_df['few_shot_prompt'] = prompts
        val_df.to_csv(OUTPUT_PATH / f'{language}_val_ocr_fsprompts.csv', index=False)


if __name__=='__main__':
    precalculate_for_original_ocr()
    # concat_ocr_files_by_language_original_ocr()
