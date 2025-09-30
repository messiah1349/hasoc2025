import logging
import pandas as pd
from src.common.logging import setup_logging
from src.common.constants import BODO_TEST_IMAGES_PATH, BANGLA_TEST_IMAGES_PATH, OUTPUT_PATH, TRAIN_PATH
from src.common.constants import BODO_TRAIN_IMAGES_PATH, BANGLA_TRAIN_IMAGES_PATH, GUJARATI_TRAIN_IMAGES_PATH, HINDI_TRAIN_IMAGES_PATH
from src.common.constants import BODO_TEST_IMAGES_PATH, BANGLA_TEST_IMAGES_PATH, GUJARATI_TEST_IMAGES_PATH, HINDI_TEST_IMAGES_PATH
from src.gemini.gemini import GeminiCaller
from src.gemini.gemini_score import score_top_n_images_per_folder, score_test_images_every_lang


languages = ['Bangla', 'Bodo', 'Gujarati', 'Hindi']


def main():
    # gemini_caller = GeminiCaller(return_type='ocr', model='gemini-2.5-flash')
    # score_top_n_images_of_every_lang(gemini_caller, n=10, output_prefix='test10ocr_2p5_flash')

    # gemini_caller = GeminiCaller(return_type='meme_output', model='gemini-2.5-flash-lite-preview-06-17')
    # score_test_images_every_lang(gemini_caller
    #     , output_prefix='eval_lite_v1'
    #     # , image_folders=[BODO_TEST_IMAGES_PATH],
    # )

    # gemini_caller = GeminiCaller(return_type='meme_output', model='gemini-2.5-flash')
    # score_test_images_every_lang(
    #     gemini_caller,
    #     output_prefix='eval_full_v2',
    #     image_folders=None,
    #     n=None,
    # )
    # score_test_images_every_lang(
    #     gemini_caller,
    #     output_prefix='eval_full_v2',
    #     image_folders=[BANGLA_TEST_IMAGES_PATH],
    #     n=None,
    # )
    # gemini_caller = GeminiCaller(return_type='ocr', model='gemini-2.5-flash-lite-preview-06-17')
    # score_test_images_every_lang(
    #     gemini_caller,
    #     output_prefix='ocr_full_train',
    #     image_folders=[BODO_TRAIN_IMAGES_PATH, GUJARATI_TRAIN_IMAGES_PATH, HINDI_TRAIN_IMAGES_PATH, BANGLA_TRAIN_IMAGES_PATH],
    #     n=None,
    # )
    # score_test_images_every_lang(
    #     gemini_caller,
    #     output_prefix='ocr_full_test',
    #     image_folders=[BODO_TEST_IMAGES_PATH, GUJARATI_TEST_IMAGES_PATH, HINDI_TEST_IMAGES_PATH, BANGLA_TEST_IMAGES_PATH],
    #     n=None,
    # )
    # xlm_results = pd.read_csv(TRAIN_PATH / 'valid_df_0.csv')
    # xlm_ids = [id for id in xlm_results['image_id']]
    # # print(xlm_ids)
    # gemini_caller = GeminiCaller(return_type='meme_output', model='gemini-2.5-flash')
    # score_top_n_images_per_folder(
    #     gemini_caller,
    #     output_prefix='flash_to_check_eval',
    #     image_folders=[BODO_TRAIN_IMAGES_PATH, GUJARATI_TRAIN_IMAGES_PATH, HINDI_TRAIN_IMAGES_PATH, BANGLA_TRAIN_IMAGES_PATH],
    #     n=None,
    #     filter_image_names=xlm_ids,
    # )

    gemini_caller = GeminiCaller(return_type='fs_call', model='gemini-2.5-flash')
    # version = 'v1'
    # for language in languages:
    # for language in ['Gujarati', 'Hindi', 'Bangla']:
    #     test_df = pd.read_csv(OUTPUT_PATH / f'{language}_test_fsprompts.csv')
    #     gemini_output_test = gemini_caller.call_fs_prompts(test_df)
    #     gemini_output_test.to_csv(OUTPUT_PATH / f"{language}_test_fsp_gemini_scores_{version}.csv", index=False)
    #
    #     val_df = pd.read_csv(OUTPUT_PATH / f'{language}_val_fsprompts.csv')
    #     gemini_output_val = gemini_caller.call_fs_prompts(val_df)
    #     gemini_output_val.to_csv(OUTPUT_PATH / f"{language}_val_fsp_gemini_scores_{version}.csv", index=False)

    version = 'v2'
    for language in ['Gujarati', 'Hindi', 'Bangla']:
    # for language in ['Bodo']:
        test_df = pd.read_csv(OUTPUT_PATH / f'{language}_test_ocr_fsprompts.csv')
        logger.warn(f'call for language {language}')
        gemini_output_test = gemini_caller.call_fs_prompts(test_df)
        gemini_output_test.to_csv(OUTPUT_PATH / f"{language}_test_ocr_fsp_gemini_scores_{version}.csv", index=False)

        val_df = pd.read_csv(OUTPUT_PATH / f'{language}_val_ocr_fsprompts.csv')
        gemini_output_val = gemini_caller.call_fs_prompts(val_df)
        gemini_output_val.to_csv(OUTPUT_PATH / f"{language}_val_ocr_fsp_gemini_scores_{version}.csv", index=False)
if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger(__name__)
    main()
