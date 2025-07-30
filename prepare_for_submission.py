from pathlib import Path
import pandas as pd
from src.common.constants import OUTPUT_PATH, TEST_PATH

languages = ['Bangla', 'Bodo', 'Gujarati', 'Hindi']

columns_rename = {
    'id': 'Ids',
    'sentiment': 'Sentiment',
    'sarcasm': 'Sarcasm',
    'vulgar': 'Vulgar',
    'abuse': 'Abuse'
}

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

fillna_values = {
    'Sentiment': 'Negative',
    'Sarcasm': 'Sarcastic',
    'Vulgar': 'Non Vulgar',
    'Abuse': 'Abusive',
}

for language in languages:
    # output_file = OUTPUT_PATH / f"{language}_test_images_flash_to_check_eval.csv"
    output_file = OUTPUT_PATH / f"{language}_train_images_flash_to_check_eval.csv"
    df = pd.read_csv(output_file)
    df = df.rename(columns=columns_rename)
    df = df[columns_rename.values()]
    for column, replace_values_dict in values_rename.items():
        df[column] = df[column].replace(replace_values_dict)

    new_file = OUTPUT_PATH / f"{language}_check_eval.csv"
    # new_file = OUTPUT_PATH / f"{language}_to_submit_gemini.csv"
    # curr_test_path = TEST_PATH / f"{language.capitalize()}_test_2025" / f"{language.capitalize()}_test_data_wo_label.csv"
    # test_samples = pd.read_csv(curr_test_path)[['Ids']]
    # test_samples['order'] = range(len(test_samples))
    # df = test_samples.merge(df, how='left').sort_values('order').drop('order', axis=1).fillna(fillna_values)
    df.to_csv(new_file, index=False)

