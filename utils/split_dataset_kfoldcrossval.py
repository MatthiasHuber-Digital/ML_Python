import os
from pathlib import Path
import random
import shutil
from sklearn.model_selection import KFold
from tqdm import tqdm
from utils.file_and_path_operations import find_files_of_extension_binned


def conduct_train_val_test_split_incl_file_copying(
    path_source_data: str,
    path_dest_data: str,
    k_folds: int,
    image_extensions: list[str] = ["jpg", "jpeg", "png", "bmp"],
    frac_train_and_val_data: float = 0.9,
) -> dict:
    """This function takes an image dataset (classification, i.e. no label files) and conducts a train-test split with subsequent k-fold splits.

    The algorithm copies the image data into newly created output folders for test and k-fold-folders for training and validation.
    A dictionary with the respective folder paths is returned in order to feed the pytorch dataset/dataloader objects.

    Args:
        path_source_data (str): Path of the source data folder (contains training and test images). The images need to be in subfolders carrying their class name.
        path_dest_data (str): Path of the destination folder. Attention: any existing data will be overwritten.
        k_folds (int): Number of folds to be created.
        image_extensions (list[str], optional): File extensions (without "."!) of the images to be sought for. Defaults to ["jpg", "jpeg", "png", "bmp"].
        frac_train_and_val_data (float, optional): Share of training and validation data. Defaults to 0.8.

    Raises:
        ValueError: If the data fraction factor is not in the permitted range.

    Returns:
        dict: A dictionary containing the folder paths for test and k-fold training/validation folders.
    """
    if not (frac_train_and_val_data < 1.0 and frac_train_and_val_data > 0.0):
        raise ValueError("The split fraction factor must be between 0 and 1.")
    path_test_dest_folder = Path(path_dest_data, "test")
    dict_image_filepaths_by_class = find_files_of_extension_binned(
        path_root_dir=path_source_data,
        list_sought_extensions=image_extensions,
        return_relative_paths_only=False,
    )

    random.seed(42)
    print("Iterating over classes...")
    for idx_bin, (bin, list_class_paths) in tqdm(enumerate(dict_image_filepaths_by_class.items())):
        random.shuffle(list_class_paths)  # shuffling happens INPLACE - no assignment needed

        number_class_samples = len(list_class_paths)

        train_size_and_val_size = int(number_class_samples * frac_train_and_val_data)
        test_size = number_class_samples - train_size_and_val_size

        class_name = bin.split("/")[-1]

        list_test_filepaths = list_class_paths[:test_size]
        list_train_val_filepaths = list_class_paths[test_size:]

        def copy_test_set_to_destination(
            path_test_dest_folder,
            class_name,
            list_test_filepaths,
        ):
            path_test_dest_folder_class = Path(path_test_dest_folder) / Path(
                class_name
            )  # os.path.join(path_test_dest_folder, class_name)
            path_test_dest_folder_class.mkdir(exist_ok=True, parents=True)

            dest_folder = path_test_dest_folder_class
            print("Copying test files...")
            for path_src_file in list_test_filepaths:
                filename = path_src_file.split("/")[-1]
                shutil.copy(path_src_file, os.path.join(dest_folder, filename))

        copy_test_set_to_destination(
            path_test_dest_folder=path_test_dest_folder,
            class_name=class_name,
            list_test_filepaths=list_test_filepaths,
        )

        def copy_train_val_sets_to_destinations(
            path_dest_data: str | Path,
            class_name: str,
            list_train_val_filepaths: list,
            number_folds: int,
            first_class: bool,
        ):
            list_paths_train_and_val_folders = list()

            kf = KFold(n_splits=number_folds)
            kf.get_n_splits(list_train_val_filepaths)
            for idx_fold, (list_fold_train_indices, list_fold_val_indices) in enumerate(
                kf.split(list_train_val_filepaths)
            ):
                path_train_fold_dest_folder = Path(path_dest_data) / Path(
                    "fold_" + str(idx_fold) + "/train/"
                )
                path_val_fold_dest_folder = Path(path_dest_data) / Path(
                    "fold_" + str(idx_fold) + "/val/"
                )
                if first_class:
                    # Creating parent folders once
                    path_train_fold_dest_folder.mkdir(exist_ok=True, parents=True)
                    path_val_fold_dest_folder.mkdir(exist_ok=True, parents=True)

                path_class_train_fold_dest_folder = path_train_fold_dest_folder / Path(class_name)
                path_class_train_fold_dest_folder.mkdir(exist_ok=False, parents=False)

                path_class_val_fold_dest_folder = path_val_fold_dest_folder / Path(class_name)
                path_class_val_fold_dest_folder.mkdir(exist_ok=False, parents=False)

                list_fold_train_paths = [
                    f
                    for i, f in enumerate(list_train_val_filepaths)
                    if i in list_fold_train_indices
                ]
                list_fold_val_paths = [
                    f for i, f in enumerate(list_train_val_filepaths) if i in list_fold_val_indices
                ]

                print("Copying train images for fold: " + str(idx_fold))
                for path_train_src_file in list_fold_train_paths:
                    filename = path_train_src_file.split("/")[-1]
                    shutil.copy(
                        path_train_src_file,
                        Path(path_class_train_fold_dest_folder) / Path(filename),
                    )

                print("Copying val images for fold: " + str(idx_fold))
                for path_val_src_file in list_fold_val_paths:
                    filename = path_val_src_file.split("/")[-1]
                    shutil.copy(
                        path_val_src_file,
                        Path(path_class_val_fold_dest_folder) / Path(filename),
                    )

                list_paths_train_and_val_folders.append(
                    {
                        "fold": idx_fold,
                        "train": path_class_train_fold_dest_folder,
                        "val": path_class_val_fold_dest_folder,
                    }
                )

            return list_paths_train_and_val_folders

        list_paths_train_and_val_folders = copy_train_val_sets_to_destinations(
            path_dest_data=path_dest_data,
            class_name=class_name,
            list_train_val_filepaths=list_train_val_filepaths,
            number_folds=k_folds,
            first_class=True if idx_bin == 0 else False,
        )

    print("All finished.")
    return {
        "test": path_test_dest_folder,
        "train_and_val": list_paths_train_and_val_folders,
    }


def get_kfold_directory_list(
    path_image_dir: str,
) -> list[str]:
    """This function returns a list of the k-folds of a kfold cross-validation split.

    Args:
        path_image_dir (str): The path into which the k-folds were stored.

    Returns:
        list[str]: List of the k-folds of the data split.
    """
    list_kfold_dirs = []
    for dir, _, _ in os.walk(path_image_dir):
        if dir.startswith("fold_"):
            list_kfold_dirs.append(
                {
                    "idx_fold": int(dir.split("_")[-1]),
                    "train": dir + "/train",
                    "val": dir + "/val",
                }
            )
    return list_kfold_dirs


if __name__ == "__main__":
    path_source_data = "/home/matthias/workspace/Coding/00_vista_medizina/00_data/2024-11-08/kaggle_bone_fracture_detection_small_2024-11-08/bf_orig_bgr_classification_2cl/unsplit"
    path_dest_data = "/home/matthias/workspace/Coding/00_vista_medizina/00_data/2024-11-08/kaggle_bone_fracture_detection_small_2024-11-08/bf_orig_bgr_classification_2cl/kfold"
    dict_split_results = conduct_train_val_test_split_incl_file_copying(
        path_source_data=path_source_data,
        path_dest_data=path_dest_data,
        k_folds=5,
    )
    print(dict_split_results)
