from PIL import Image
import logging
from src.gemini.gemini import GeminiCaller
from src.common.constants import TRAIN_PATH, OUTPUT_PATH, BANGLA_TRAIN_IMAGES_PATH, BODO_TRAIN_IMAGES_PATH, HINDI_TRAIN_IMAGES_PATH, GUJARATI_TRAIN_IMAGES_PATH
from src.common.logging import setup_logging


def score_top_n_images_of_every_lang(gemini_caller: GeminiCaller, n=50, output_prefix: str='top_50'):

    for train_images_path in [
            BODO_TRAIN_IMAGES_PATH, 
            HINDI_TRAIN_IMAGES_PATH, 
            GUJARATI_TRAIN_IMAGES_PATH, 
            BANGLA_TRAIN_IMAGES_PATH, 
        ]:
        output_file_name = train_images_path.name
        # logger.warning(f"language = {output_file_name}")
        train_images_pathes = [path for path in train_images_path.glob('*')][:n]
        train_ids = [image.name for image in train_images_pathes]
        train_images = []
        for image_path in train_images_pathes:
            with Image.open(image_path) as img:
                img.load()
                train_images.append(img.copy())
        response_df = gemini_caller.call_several_images(train_images, train_ids)
        # print(response_df)
        response_df.to_csv(OUTPUT_PATH / f'{output_file_name}_{output_prefix}', index=False)

def main():
    # gemini_caller = GeminiCaller(return_type='ocr')
    # score_top_n_images_of_every_lang(gemini_caller, n=10, output_prefix='test10ocr')

    gemini_caller = GeminiCaller(return_type='ocr', model='gemini-2.5-flash')
    score_top_n_images_of_every_lang(gemini_caller, n=10, output_prefix='test10ocr_2p5_flash')



if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger(__name__)
    main()
