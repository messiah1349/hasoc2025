from pathlib import Path
import pandas as pd
from src.common.constants import OUTPUT_PATH, TEST_PATH, TRAIN_PATH

languages = ['Bangla', 'Bodo', 'Gujarati', 'Hindi']

columns_rename = {
    'id': 'Ids',
    'sentiment': 'Sentiment',
    'sarcasm': 'Sarcasm',
    'vulgar': 'Vulgar',
    'abuse': 'Abuse'
}

xlm_results = pd.read_csv(TRAIN_PATH / 'valid_df_0.csv')

all_lang_train_df = pd.DataFrame()

for language in languages:
    lang_train_path = TRAIN_PATH / f"{language.capitalize()}_train_2025" / f"{language.capitalize()}_train_2025.csv"
    lang_train_df = pd.read_csv(lang_train_path)
    lang_train_df['language'] = language
    all_lang_train_df = pd.concat([all_lang_train_df, lang_train_df])

eval_df = all_lang_train_df[all_lang_train_df['Ids'].isin(xlm_results['image_id'])]
eval_df.to_csv(TRAIN_PATH / eval_df)

