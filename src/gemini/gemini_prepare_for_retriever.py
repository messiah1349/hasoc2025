import pandas as pd
from src.common.constants import OUTPUT_PATH, TRAIN_PATH, TEST_PATH


values_rename = {
    'Sentiment': {
        'negative': 'Negative',
        'positive': 'Positive',
        'neutral': 'Neutral',
    },
    'Sarcasm': {
        False: 'Non-Sarcastic',
        True: 'Sarcastic',
    },
    'Vulgar': {
        False: 'Non Vulgar',
        True: 'Vulgar',
    },
    'Abuse': {
        False: 'Non-abusive',
        True: 'Abusive',
    },
}


def concat_texts(df: pd.DataFrame) -> pd.DataFrame:
    """
        df contains columns: 
            location, image_description, translated_text, id
        one id may be in several row
        function return dataframe with concated image_description and translated_text. One row - one id.
        output columns:
            id
            image_descripiton: with format: "location_value_row1: image_description_value_row1; location_value_row2: image_description_value_row2; ...
            translated_text: with format: "location_value_row1: translated_text_value_row1; location_value_row2: translated_text_value_row2; ...
    """
    grouped = df.groupby('id')
    concatenated_data = []

    for id_value, group in grouped:
        image_description_concat = '; '.join(
            f"{row['location']}: {row['image_description']}".replace('\n', ' ') for _, row in group.iterrows()
        )
        translated_text_concat = '; '.join(
            f"{row['location']}: {row['translated_text']}".replace('\n', ' ') for _, row in group.iterrows()
        )
        concatenated_data.append({
            'id': id_value,
            'image_description': image_description_concat,
            'translated_text': translated_text_concat
        })

    return pd.DataFrame(concatenated_data).rename(columns={'id': 'Ids'})
        
def concat_ocr_files_by_language():

    languages = ['Bangla', 'Bodo', 'Gujarati', 'Hindi']
    target_columns = ['Sentiment','Sarcasm','Vulgar','Abuse']
    xlm_results = pd.read_csv(TRAIN_PATH / 'valid_df_0.csv')

    for language in languages:
        train_output = pd.read_csv(OUTPUT_PATH / f"{language.capitalize()}_train_images_ocr_full_train.csv")
        test_output = pd.read_csv(OUTPUT_PATH / f"{language.capitalize()}_test_images_ocr_full_test.csv")

        lang_train_path = TRAIN_PATH / f"{language.capitalize()}_train_2025" / f"{language.capitalize()}_train_data.csv"
        train_with_labels = pd.read_csv(lang_train_path)

        all_train_concated = concat_texts(train_output)
        all_train_concated = all_train_concated.merge(train_with_labels, how='left')
        for target_column in target_columns:
            val_ren_dict = values_rename[target_column]
            val_ren_dict = {val: key for key, val in val_ren_dict.items()}
            all_train_concated[target_column] = all_train_concated[target_column].replace(val_ren_dict)
        all_train_concated['ground_truth'] = all_train_concated[target_columns].to_dict(orient='records')
        val_concated = all_train_concated[all_train_concated['Ids'].isin(xlm_results['image_id'])]
        train_concated = all_train_concated[~all_train_concated['Ids'].isin(xlm_results['image_id'])]

        test_concated = concat_texts(test_output)

        train_concated.to_csv(OUTPUT_PATH / f"{language.capitalize()}_train_ocr_concated.csv", index=False)
        val_concated.to_csv(OUTPUT_PATH / f"{language.capitalize()}_val_ocr_concated.csv", index=False)
        test_concated.to_csv(OUTPUT_PATH / f"{language.capitalize()}_test_ocr_concated.csv", index=False)

def concat_ocr_files_by_language_original_ocr():

    languages = ['Bangla', 'Bodo', 'Gujarati', 'Hindi']
    target_columns = ['Sentiment','Sarcasm','Vulgar','Abuse']
    xlm_results = pd.read_csv(TRAIN_PATH / 'valid_df_0.csv')

    for language in languages:
        # train_output = pd.read_csv(TRAIN_PATH / f"{language}_train_2025" / f"{language.capitalize()}_train_data.csv")
        test_output = pd.read_csv(TEST_PATH / f"{language}_test_2025" / f"{language.capitalize()}_test_data_wo_label.csv")

        lang_train_path = TRAIN_PATH / f"{language.capitalize()}_train_2025" / f"{language.capitalize()}_train_data.csv"
        train_with_labels = pd.read_csv(lang_train_path)

        all_train_concated = train_with_labels.copy()
        # all_train_concated = all_train_concated.merge(train_with_labels, how='left')
        for target_column in target_columns:
            val_ren_dict = values_rename[target_column]
            val_ren_dict = {val: key for key, val in val_ren_dict.items()}
            all_train_concated[target_column] = all_train_concated[target_column].replace(val_ren_dict)
        all_train_concated['ground_truth'] = all_train_concated[target_columns].to_dict(orient='records')
        val_concated = all_train_concated[all_train_concated['Ids'].isin(xlm_results['image_id'])]
        train_concated = all_train_concated[~all_train_concated['Ids'].isin(xlm_results['image_id'])]

        test_concated = test_output.copy()

        train_concated.to_csv(OUTPUT_PATH / f"{language.capitalize()}_train_original_ocr.csv", index=False)
        val_concated.to_csv(OUTPUT_PATH / f"{language.capitalize()}_val_original_ocr.csv", index=False)
        test_concated.to_csv(OUTPUT_PATH / f"{language.capitalize()}_test_original_ocr.csv", index=False)
