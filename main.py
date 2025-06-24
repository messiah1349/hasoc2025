from PIL import Image
from src.gemini.gemini import GeminiCaller
from src.common.constants import TRAIN_PATH, OUTPUT_PATH

def main():
    train_images_pathes = [path for path in TRAIN_PATH.glob('*')]
    train_ids = [image.name for image in train_images_pathes]
    train_images = [Image.open(image_path) for image_path in train_images_pathes]
    print(f"{train_images=}")
    gemini_caller = GeminiCaller()
    response_df = gemini_caller.call_several_images(train_images, train_ids)
    response_df.to_csv(OUTPUT_PATH / 'scores_5_test_screenshots.csv', index=False)


if __name__ == "__main__":
    main()
