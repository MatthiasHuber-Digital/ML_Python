import os


def find_files_of_extension(
    path_root_dir: str,
    list_sought_extensions: list,
    return_relative_paths_only: bool = False,
) -> list:
    """This function finds all text files in the given directory and its subdirectories.

    Args:
        path_root_dir (str): The root directory in which to search.
        list_sought_extensions (list): The file extensions to search for. NO "." should be included!

    Returns:
        list_paths_files (list): A list of the filepaths which were found.
    """
    list_paths_files = []
    for dirpath, _, filenames in os.walk(path_root_dir):

        for filename in filenames:

            if filename.split(".")[-1] in list_sought_extensions:
                if return_relative_paths_only:
                    list_paths_files.append(
                        os.path.relpath(os.path.join(dirpath, filename), path_root_dir)
                    )
                else:
                    list_paths_files.append(os.path.join(dirpath, filename))

    return list_paths_files


def find_files_of_extension_binned(
    path_root_dir: str,
    list_sought_extensions: list,
    return_relative_paths_only: bool = False,
) -> list:
    """This function finds all text files in the given directory and its subdirectories.

    Args:
        path_root_dir (str): The root directory in which to search.
        list_sought_extensions (list): The file extensions to search for. NO "." should be included!

    Returns:
        list_paths_files (list): A list of the filepaths which were found.
    """
    dict_filespaths_binned = dict()
    for dirpath, _, filenames in os.walk(path_root_dir):

        list_paths_files = []
        for filename in filenames:

            if filename.split(".")[-1] in list_sought_extensions:
                if return_relative_paths_only:
                    list_paths_files.append(
                        os.path.relpath(os.path.join(dirpath, filename), path_root_dir)
                    )
                else:
                    list_paths_files.append(os.path.join(dirpath, filename))

        if len(list_paths_files) > 0 and dirpath is not None:
            # dict_filespaths_binned = {dirpath: list_paths_files}
            # else:
            dict_filespaths_binned.update({dirpath.split("/")[-1]: list_paths_files})

    return dict_filespaths_binned


def get_label_and_image_paths(path_images_dir: str, path_labels_dir: str) -> dict[list]:
    """This function returns a dict of two lists, one for the image and the other for the label paths.

    Args:
        path_images_dir (str): Path of the image files' directory.
        path_labels_dir (str): Path of the label files' directory.

    Returns:
        dict[list]: Dict of lists for the label and the image paths.
    """
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
