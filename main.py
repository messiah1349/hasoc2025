import logging
from src.common.logging import setup_logging
from src.common.constants import BODO_TEST_IMAGES_PATH, BANGLA_TEST_IMAGES_PATH
from src.common.constants import BODO_TRAIN_IMAGES_PATH, BANGLA_TRAIN_IMAGES_PATH, GUJARATI_TRAIN_IMAGES_PATH, HINDI_TRAIN_IMAGES_PATH
from src.gemini.gemini import GeminiCaller
from src.gemini.gemini_score import score_top_n_images_per_folder, score_test_images_every_lang



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
    gemini_caller = GeminiCaller(return_type='ocr', model='gemini-2.5-flash-lite-preview-06-17')
    score_test_images_every_lang(
        gemini_caller,
        output_prefix='ocr_full_train',
        image_folders=[BODO_TRAIN_IMAGES_PATH, GUJARATI_TRAIN_IMAGES_PATH, HINDI_TRAIN_IMAGES_PATH, BANGLA_TRAIN_IMAGES_PATH],
        n=None,
    )

if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger(__name__)
    main()
