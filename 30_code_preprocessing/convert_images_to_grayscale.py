import cv2
import os
from tqdm import tqdm


def find_image_files(root_dir):
    image_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png') or filename.endswith('.bmp') or filename.endswith('.tif'):
                image_files.append(os.path.join(dirpath, filename))
    return image_files


def convert_images_to_grayscale(list_paths_images: list):
    for image_path in tqdm(list_paths_images):
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(image_path, gray_image)


if __name__ == '__main__':
    path_image_dir = '/home/matthias/workspace/Coding/00_vista_medizina/00_data/2024-11-08/kaggle_bone_fracture_detection_small_2024-11-08/bone_fracture_detection.v4-v4.yolov8_grayscale'

    list_image_files = find_image_files(root_dir=path_image_dir)

    convert_images_to_grayscale(list_paths_images=list_image_files)