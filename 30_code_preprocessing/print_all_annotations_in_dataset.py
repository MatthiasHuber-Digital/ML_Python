import os
from tqdm import tqdm
from copy import deepcopy


def find_files_of_extension(path_root_dir: str, extension: str) -> list:
    """This function finds all text files in the given directory and its subdirectories.

    Args:
        path_root_dir (str): The root directory in which to search.
        extension (str): The file extension to search for. NO "." should be included!

    Returns:
        list_paths_files (list): A list of the filepaths which were found.
    """
    list_paths_files = []
    for dirpath, _, filenames in os.walk(path_root_dir):
        
        for filename in filenames:

            if filename.endswith('.' + extension):
                list_paths_files.append(os.path.join(dirpath, filename))

    return list_paths_files


def change_annotation_filenames_in_dataset(path_ds: str):
    """This function prints out all annotations in the entire dataset.

    Args:
        path_ds (str): This is the path to the dataset directory.
    """
    list_txt_files = find_files_of_extension(path_root_dir=path_ds, extension='txt')
    save_output = False
    
    for txt_file in tqdm(list_txt_files):
        
        with open(txt_file, 'r') as input_file:
            lines = input_file.readlines()

            if len(lines) > 1:
                save_output = True
        
        if save_output:
            output_file = deepcopy(txt_file).replace("._rectangular", "")
            os.rename(txt_file, output_file)


def print_all_annotations_in_dataset(path_ds: str):
    """This function prints out all annotations in the entire dataset.

    Args:
        path_ds (str): This is the path to the dataset directory.
    """
    list_txt_files = find_files_of_extension(path_root_dir=path_ds, extension='txt')

    for txt_file in tqdm(list_txt_files):
        with open(txt_file, 'r') as input_file, open(txt_file.replace("._rectangular", ""), 'w') as output_file:
            lines = input_file.readlines()

            if len(lines) > 1:
                print(lines)


if __name__ == '__main__':
    change_annotation_filenames_in_dataset(path_ds='/home/matthias/workspace/Coding/00_vista_medizina/00_data/2024-11-21/single_class_all_categories')
