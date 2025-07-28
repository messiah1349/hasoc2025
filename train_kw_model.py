from src.kw_model.kw_model import TextClassifier
from pathlib import Path
import pandas as pd
from src.common.constants import OUTPUT_PATH, TEST_PATH, TRAIN_PATH

languages = ['Bangla', 'Bodo', 'Gujarati', 'Hindi']

for language in languages:
    curr_test_path = TEST_PATH / f"{language.capitalize()}_test_2025" / f"{language.capitalize()}_test_data_wo_label.csv"
    test_samples = pd.read_csv(curr_test_path)[['Ids']]

    curr_train_path = TRAIN_PATH / f"{language.capitalize()}_train_2025" / f"{language.capitalize()}_train_data.csv"
    train_samples = pd.read_csv(curr_train_path)[['Ids']]

