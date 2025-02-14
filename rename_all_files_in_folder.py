import os
from pathlib import Path
from tqdm import tqdm


def rename_files_in_folder(path_folder):
    """This function renames all files in the folder and assigns as prefix of their name the folder name."""
    folder_name = path_folder.split("/")[-1]

    for filename in tqdm(os.listdir(path_folder)):

        old_file_path = Path(path_folder) / Path(filename)
        new_file_path = Path(path_folder) / Path(folder_name + filename)  # "_" + filename)

        if os.path.isfile(old_file_path):
            # new_filename = f"{Path(filepath).stem}_{path_folder.name}"
            os.rename(old_file_path, new_file_path)


def rename_files_in_folder_custom_prefix(path_folder, str_prefix):
    """This function renames all files in the folder and assigns as prefix of their name the provided string."""
    for filename in tqdm(os.listdir(path_folder)):

        old_file_path = Path(path_folder) / Path(filename)
        new_file_path = Path(path_folder) / Path(str_prefix + "_" + filename)

        if os.path.isfile(old_file_path):
            os.rename(old_file_path, new_file_path)


if __name__ == "__main__":
    rename_files_in_folder(
        "/home/matthias/workspace/Coding/20_helmet_reco/data_workplace_safety_trainpics/train/train_unsafe_101-200/train_unsafe_101-200_images"
    )
    rename_files_in_folder_custom_prefix(
        "/home/matthias/workspace/Coding/20_helmet_reco/data_workplace_safety_trainpics/train/train_unsafe_101-200/train_unsafe_101-200_labels",
        str_prefix="train_unsafe_101-200_images",
    )
