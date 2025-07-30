import pandas as pd
from src.common.constants import OUTPUT_PATH, TRAIN_PATH


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
            f"{row['location']}: {row['image_description']}" for _, row in group.iterrows()
        )
        translated_text_concat = '; '.join(
            f"{row['location']}: {row['translated_text']}" for _, row in group.iterrows()
        )
        concatenated_data.append({
            'id': id_value,
            'image_description': image_description_concat,
            'translated_text': translated_text_concat
        })

    return pd.DataFrame(concatenated_data).rename(columns={'id': 'Ids'})
        
def concat_ocr_files_by_language():

    languages = ['Bangla', 'Bodo', 'Gujarati', 'Hindi']

    for language in languages:
        train_output = pd.read_csv(OUTPUT_PATH / f"{language.capitalize()}_train_images_ocr_full_train.csv")
        test_output = pd.read_csv(OUTPUT_PATH / f"{language.capitalize()}_test_images_ocr_full_test.csv")

        lang_train_path = TRAIN_PATH / f"{language.capitalize()}_train_2025" / f"{language.capitalize()}_train_data.csv"
        train_with_labels = pd.read_csv(lang_train_path)

        train_concated = concat_texts(train_output)
        train_concated = train_concated.merge(train_with_labels, how='left')
        test_concated = concat_texts(test_output)

        train_concated.to_csv(OUTPUT_PATH / f"{language.capitalize()}_train_ocr_concated.csv", index=False)
        test_concated.to_csv(OUTPUT_PATH / f"{language.capitalize()}_test_ocr_concated.csv", index=False)
