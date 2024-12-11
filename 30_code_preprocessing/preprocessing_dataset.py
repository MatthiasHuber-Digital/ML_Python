from utils.image_manipulation import preprocess_dataset, OptionsPrepContrEnhancement
from pathlib import Path
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch
from PIL import Image

"""class OptionsPrepContrEnhancement(str, Enum):
    HISTO_EQUAL = "Histogram equalization."
    CLAHE = "CLAHE contrast enhancement."
    UNSHARP_MASKING = (
        "Unsharp masking is superponed onto the picture, which helps w.r.t. contrast and sharpness."
    )
    MEIJERING = "Meijering filter."
    SATO = "SATO filter contrast enhancement."
    SCHARR = "Scharr contrast enhancement."
    HESSIAN = "Hessian contrast enhancement."""

if __name__ == "__main__":
    path_images = "/home/matthias/workspace/Coding/00_vista_medizina/00_data/2024-11-08/kaggle_bone_fracture_detection_small_2024-11-08/bf_orig_bgr_poly"
    path_base_output = Path(
        "/home/matthias/workspace/Coding/00_vista_medizina/00_data/2024-11-08/kaggle_bone_fracture_detection_small_2024-11-08/bf_histo_bgr_poly"
    )
    """
    file_path = "/home/matthias/workspace/Coding/00_vista_medizina/00_data/2024-11-08/kaggle_bone_fracture_detection_small_2024-11-08/bf_orig_bgr_poly"

    sam = sam_model_registry["vit_h"](checkpoint="SAM/pretrained_weights/SAM_weights.pth")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    import numpy as np

    selected_image = Image.open(file_path)

    np_selected_image = np.array(selected_image)
    np_selected_image

    masks = mask_generator.generate(np_selected_image)
    """

    suffix = ""
    path_output = path_base_output / Path(suffix)
    filters = [OptionsPrepContrEnhancement.HISTO_EQUAL]
    preprocess_dataset(
        path_image_dir=path_images,
        list_filters_to_apply=filters,
        path_image_output_dir=path_output,
        image_output_filename_suffix=suffix,
        overwrite_images=False,
    )
    """
    suffix = "HISTO_EQUAL"
    path_output = path_base_output / Path(suffix)
    filters = [OptionsPrepContrEnhancement.HISTO_EQUAL]
    preprocess_dataset(
        path_image_dir=path_images,
        list_filters_to_apply=filters,
        path_image_output_dir=path_output,
        image_output_filename_suffix=suffix,
        overwrite_images=False,
    )
    suffix = "CLAHE"
    path_output = path_base_output / Path(suffix)
    filters = [OptionsPrepContrEnhancement.CLAHE]
    preprocess_dataset(
        path_image_dir=path_images,
        list_filters_to_apply=filters,
        path_image_output_dir=path_output,
        image_output_filename_suffix=suffix,
        overwrite_images=False,
    )

    suffix = "UNSHARP_MASKING"
    path_output = path_base_output / Path(suffix)
    filters = [OptionsPrepContrEnhancement.UNSHARP_MASKING]
    preprocess_dataset(
        path_image_dir=path_images,
        list_filters_to_apply=filters,
        path_image_output_dir=path_output,
        image_output_filename_suffix=suffix,
        overwrite_images=False,
    )

    suffix = "MEIJERING_HISTO"
    path_output = path_base_output / Path(suffix)
    filters = [OptionsPrepContrEnhancement.MEIJERING, OptionsPrepContrEnhancement.HISTO_EQUAL]
    preprocess_dataset(
        path_image_dir=path_images,
        list_filters_to_apply=filters,
        path_image_output_dir=path_output,
        image_output_filename_suffix=suffix,
        overwrite_images=False,
    )

    suffix = "SATO_HISTO"
    path_output = path_base_output / Path(suffix)
    filters = [OptionsPrepContrEnhancement.SATO, OptionsPrepContrEnhancement.HISTO_EQUAL]
    preprocess_dataset(
        path_image_dir=path_images,
        list_filters_to_apply=filters,
        path_image_output_dir=path_output,
        image_output_filename_suffix=suffix,
        overwrite_images=False,
    )

    suffix = "SCHARR_HISTO"
    path_output = path_base_output / Path(suffix)
    filters = [OptionsPrepContrEnhancement.SCHARR, OptionsPrepContrEnhancement.HISTO_EQUAL]
    preprocess_dataset(
        path_image_dir=path_images,
        list_filters_to_apply=filters,
        path_image_output_dir=path_output,
        image_output_filename_suffix=suffix,
        overwrite_images=False,
    )

    suffix = "HESSIAN_CLAHE"
    path_output = path_base_output / Path(suffix)
    filters = [OptionsPrepContrEnhancement.HESSIAN, OptionsPrepContrEnhancement.CLAHE]
    preprocess_dataset(
        path_image_dir=path_images,
        list_filters_to_apply=filters,
        path_image_output_dir=path_output,
        image_output_filename_suffix=suffix,
        overwrite_images=False,
    )
    """
