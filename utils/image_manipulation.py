import cv2
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from pathlib import Path
from skimage.filters import unsharp_mask, meijering, sato, scharr, hessian
from skimage.exposure import equalize_hist, equalize_adapthist
from tqdm import tqdm
import cv2
from enum import Enum
import numpy as np
import os
from pathlib import Path
from tqdm.auto import tqdm

from utils.file_and_path_operations import find_files_of_extension


class OptionsPrepContrEnhancement(str, Enum):
    HISTO_EQUAL = "Histogram equalization."
    CLAHE = "CLAHE contrast enhancement."
    UNSHARP_MASKING = (
        "Unsharp masking is superponed onto the picture, which helps w.r.t. contrast and sharpness."
    )
    MEIJERING = "Meijering filter."
    SATO = "SATO filter contrast enhancement."
    SCHARR = "Scharr contrast enhancement."
    HESSIAN = "Hessian contrast enhancement."


def resize_image_w_padding(image: np.ndarray, max_size: int = 224, pad: bool = False) -> np.ndarray:
    h, w = image.shape[:2]
    if h > w:
        scale = max_size / h  # Масштаб для высоты
    else:
        scale = max_size / w  # Масштаб для ширины
    new_size = (int(w * scale), int(h * scale))  # Новые размеры
    resized_image = cv2.resize(image, new_size)  # Изменяем размер изображения

    if pad:
        # Дополнение изображения до максимального размера
        padded_image = np.ones((max_size, max_size, 3), dtype=np.uint8) * 255

        half_rest_width_1 = max_size // 2 - new_size[0] // 2
        half_rest_height_1 = max_size // 2 - new_size[1] // 2

        if half_rest_width_1 == 0:
            padded_image[
                half_rest_height_1 : (new_size[1] + half_rest_height_1),
                : new_size[0],
                :,
            ] = resized_image  # 0-dim: y or height, 1-dim: x or width
        elif half_rest_height_1 == 0:
            padded_image[
                : new_size[1], half_rest_width_1 : (new_size[0] + half_rest_width_1), :
            ] = resized_image  # 0-dim: y or height, 1-dim: x or width
        else:
            raise ValueError("Unexpected situation while padding image")

        return padded_image
    else:
        return resized_image


def apply_contrast_filter_to_image(
    image_data: np.ndarray, chosen_filter: OptionsPrepContrEnhancement
) -> np.ndarray:
    """This function applies a contrast enhancement technique to the image.

    Args:
        image (np.ndarray): The image numpy data.

    Returns:
        np.ndarray: The preprocessed image data.
    """
    if chosen_filter == OptionsPrepContrEnhancement.HISTO_EQUAL:
        image_data = equalize_hist(image_data, nbins=256, mask=None)

    elif chosen_filter == OptionsPrepContrEnhancement.CLAHE:
        # image_data = equalize_adapthist(image_data, kernel_size=None, clip_limit=0.01, nbins=256)
        image_data = equalize_adapthist(image_data, kernel_size=None, clip_limit=0.01, nbins=256)

    elif chosen_filter == OptionsPrepContrEnhancement.UNSHARP_MASKING:
        image_data = unsharp_mask(image_data, radius=5, amount=2)

    elif chosen_filter == OptionsPrepContrEnhancement.MEIJERING:
        image_data = meijering(
            image_data,
            sigmas=range(1, 10, 2),
            alpha=None,
            black_ridges=True,
            mode="reflect",
            cval=0,
        )

    elif chosen_filter == OptionsPrepContrEnhancement.SATO:
        image_data = sato(
            image_data,
            sigmas=range(1, 10, 2),
            black_ridges=True,
            mode="reflect",
            cval=0,
        )

    elif chosen_filter == OptionsPrepContrEnhancement.SCHARR:
        image_data = scharr(image_data, mask=None, axis=None, mode="reflect", cval=0.0)

    elif chosen_filter == OptionsPrepContrEnhancement.HESSIAN:
        image_data = hessian(
            image_data,
            sigmas=range(1, 10, 2),
            scale_range=None,
            scale_step=None,
            alpha=0.5,
            beta=0.5,
            gamma=15,
            black_ridges=True,
            mode="reflect",
            cval=0,
        )

    else:
        raise ValueError("Unknown filter: %s" % chosen_filter)

    return image_data


def apply_canny_filter(
    image: np.ndarray,
    threshold_lower: int = 50,
    threshold_higher: int = 200,
) -> np.ndarray:
    """This function applies the canny filter to recognize the edges of a given image.

    Args:
        image (np.ndarray): Input image data.
        threshold_lower (int, optional): Lower threshold. Defaults to 50.
        threshold_higher (int, optional): Higher threshold.. Defaults to 200.

    Returns:
        np.ndarray: Edge-filtered image.
    """
    return cv2.Canny(image=image, threshold1=threshold_lower, threshold2=threshold_higher)


def apply_thresholding(image: np.ndarray, threshold: int = 50) -> np.ndarray:
    """This function applies thresholding to a given image.

    The treshold range can vary depending on the image and which output is expected.

    Args:
        image (np.ndarray): The input image data.
        threshold (int, optional): Threshold to be used. Defaults to 50.

    Returns:
        np.ndarray: Output image data.
    """

    def valueScaling(value):
        min_value = 0
        max_value = 100
        new_min = 0
        new_max = 255
        scaled_value = (value - min_value) * (new_max - new_min) / (max_value - min_value) + new_min
        return int(scaled_value)

    threshold = valueScaling(threshold)

    _, image_thresholded = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    mask = 255 - image_thresholded
    return mask
    # return cv2.bitwise_and(image, image, mask=mask)


def plot_images(plot_title: str, image: np.ndarray, image2: np.ndarray):
    """This function plots two images next to each other.

    It can be for example used in order to plot the original and the processed image.

    Args:
        plot_title (_type_): Title pf the plot.
        image (_type_): First image data.
        image2 (_type_): Second image data.
    """
    # Plot both images
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(12, 12))
    ax = axes.ravel()
    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].set_title("Original image")
    ax[1].imshow(image2, cmap=plt.cm.gray)
    ax[1].set_title(plot_title)
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    plt.show()


def preprocess_dataset(
    path_image_dir: str,
    list_filters_to_apply: list[OptionsPrepContrEnhancement],
    overwrite_images: bool = False,
    path_image_output_dir: str = None,
    image_output_filename_suffix: str = "",
):
    """This function preprocess a dataset of images (all subdirectories in directory).

    The algorithm can either overwrite the original images or save the output in another direcory.

    Args:
        path_image_dir (str): The image dataset directory.
        overwrite_images (bool, optional): If True, overwrite the original images with the preprocessed ones. Defaults to False.
        path_image_output_dir (str, optional): The output directory where to save the preprocessed images to. Defaults to None.
    """
    print("Preprocessing image dataset in path: %s" % path_image_dir)
    list_input_images = find_files_of_extension(
        path_root_dir=path_image_dir,
        list_sought_extensions=["jpg", "jpeg", "png", "bmp", "tif"],
        return_relative_paths_only=True,
    )
    print("--Found this number of files: %d" % len(list_input_images))

    print("Preprocessing files...")
    print("--Filters being applied are:")
    for filter in list_filters_to_apply:
        print("\t-- %s" % filter)

    if not overwrite_images:
        if path_image_output_dir is None:
            raise ValueError(
                "Function preprocess_dataset: path_image_output_dir was not specified."
            )
        else:
            if not os.path.exists(path_image_output_dir):
                os.makedirs(path_image_output_dir)
            else:
                print("--Warning: Output directory already exists, images may be overwritten.")

    print("--Image processing:")
    for rel_path_image in tqdm(list_input_images):

        abs_path_image = os.path.join(path_image_dir, rel_path_image)

        from skimage import io

        image_data = io.imread(abs_path_image)
        # image_data = cv2.imread(abs_path_image)

        # processed_image = cv2.normalize(
        #    image_data, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
        # )
        # processed_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2HSV)
        # plot_images(
        #    image=image_data, image2=processed_image, plot_title="Original vs. normalized image"
        # )
        # plt.pause(3)

        for filter in list_filters_to_apply:
            processed_image = apply_contrast_filter_to_image(
                image_data=image_data, chosen_filter=filter
            )

        # processed_image = cv2.cvtColor(processed_image, cv2.COLOR_HSV2BGR)

        # plot_images(
        #    image=image_data, image2=processed_image, plot_title="Original vs. Filtered image"
        # )
        # plt.pause(3)
        if overwrite_images:
            path_image_output = abs_path_image
        else:
            rel_path_dir = "/".join(rel_path_image.split("/")[:-1])
            abs_path_dir = path_image_output_dir / Path(rel_path_dir)
            if not abs_path_dir.exists():
                abs_path_dir.mkdir(parents=True)

            if len(image_output_filename_suffix) > 0:
                rel_path_split = rel_path_image.split(".")

                rel_path_image = (
                    ".".join(rel_path_split[:-1])
                    + "_"
                    + image_output_filename_suffix
                    + "."
                    + rel_path_split[-1]
                )
            path_image_output = os.path.join(path_image_output_dir, rel_path_image)

        # Method 1: only single channel image
        # processed_image = processed_image[:, :, 0]
        # processed_image = np.squeeze(processed_image, axis=2)

        # Method 2: recognize as RGB - but when saving, leads to the pic becoming unrecognizable:
        # pil_image = Image.fromarray(
        #    processed_image, "RGB"
        # )  # when saving a 3-channel image, PIL needs the image to be in uint8 (0..255) -RGB argument takes care of that
        # pil_image.save(path_image_output)

        # Method 3: convert uint8 to float so that PIL will think it is an RGB image:
        pil_image = Image.fromarray((processed_image * 255).astype(np.uint8))
        pil_image.save(path_image_output)

        # io.imsave(
        #    path_image_output, processed_image
        # )  # when saving a 3-channel image, PIL assumes the image to be in uint8 (0..255)
        # cv2.imwrite(path_image_output, processed_image)

    print("--Finished preprocessing dataset.")


def get_label_and_image_paths(path_images_dir: str, path_labels_dir: str) -> tuple[list]:
    list_paths_images = []
    for dirpath, _, filenames in os.walk(path_images_dir):

        for filename in filenames:

            if filename.endswith("." + "jpg"):
                list_paths_images.append(os.path.join(dirpath, filename))
    list_paths_images

    # list_paths_labels = []
    # for f in list_paths_images:
    #    list_paths_labels.append(".".join(f.split('.')[:-1]) + ".txt")

    list_paths_labels = []
    for dirpath, _, filenames in os.walk(path_labels_dir):

        for filename in filenames:

            if filename.endswith("." + "txt"):
                list_paths_labels.append(os.path.join(dirpath, filename))
    list_paths_labels

    return {
        "list_paths_images": list_paths_images,
        "list_paths_labels": list_paths_labels,
    }
