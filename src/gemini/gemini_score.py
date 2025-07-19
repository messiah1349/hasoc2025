from PIL import Image
from pathlib import Path
import logging
from src.gemini.gemini import GeminiCaller
from src.common.constants import OUTPUT_PATH, BANGLA_TRAIN_IMAGES_PATH, BODO_TRAIN_IMAGES_PATH, HINDI_TRAIN_IMAGES_PATH, GUJARATI_TRAIN_IMAGES_PATH
from src.common.constants import BANGLA_TEST_IMAGES_PATH, BODO_TEST_IMAGES_PATH, HINDI_TEST_IMAGES_PATH, GUJARATI_TEST_IMAGES_PATH


logger = logging.getLogger(__name__)


def score_top_n_images_per_folder(gemini_caller: GeminiCaller, image_folders: list[Path],
                                            n: int|None=50, output_prefix: str='top_50') -> None:

    if n is None:
        n = 1_000_000
    
    for image_path in image_folders:
        output_file_name = image_path.name
        logger.warning(f"path: {output_file_name}")
        train_images_pathes = [path for path in image_path.glob('*')][:n]
        train_ids = [image.name for image in train_images_pathes]
        train_images = []
        for image_path in train_images_pathes:
            with Image.open(image_path) as img:
                img.load()
                train_images.append(img.copy())
        response_df = gemini_caller.call_several_images(train_images, train_ids)
        response_df.to_csv(OUTPUT_PATH / f'{output_file_name}_{output_prefix}.csv', index=False)


def score_top_n_images_of_every_lang(gemini_caller: GeminiCaller, n: int=50, output_prefix: str='top_50') -> None:
    train_folders = [
        BODO_TRAIN_IMAGES_PATH, 
        HINDI_TRAIN_IMAGES_PATH, 
        GUJARATI_TRAIN_IMAGES_PATH, 
        BANGLA_TRAIN_IMAGES_PATH, 
    ]
    score_top_n_images_per_folder(gemini_caller, train_folders, n, output_prefix)
    

def score_test_images_every_lang(gemini_caller: GeminiCaller, output_prefix: str, n:int|None=None, image_folders: list[Path]|None=None) -> None:

    if image_folders is None:
        test_folders = [
            BODO_TEST_IMAGES_PATH, 
            HINDI_TEST_IMAGES_PATH, 
            GUJARATI_TEST_IMAGES_PATH, 
            BANGLA_TEST_IMAGES_PATH, 
        ]
    else:
        test_folders = image_folders

    score_top_n_images_per_folder(gemini_caller, test_folders, n, output_prefix)
