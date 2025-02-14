import os
from tqdm import tqdm


def find_txt_files(root_dir):
    list_txt_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".txt"):
                list_txt_files.append(os.path.join(dirpath, filename))
    return list_txt_files


def create_dicts_label_classes(path_source_label_dir: str) -> dict:
    list_source_label_files = find_txt_files(path_source_label_dir)
    dict_label_classes = {}

    for label_file in tqdm(list_source_label_files):

        label_file_descriptor = "/".join(label_file.split("/")[-3:])  # /train/labels/filename.txt

        with open(label_file, "r") as file_data:
            lines = file_data.readlines()

            labels_in_file = []
            if len(lines) > 1:
                for line in lines:
                    labels_in_file.append(int(line[0]))

            dict_label_classes.update({label_file_descriptor: labels_in_file})

    return dict_label_classes


def update_label_files_with_new_label_classes(
    path_target_label_dir: str,
    dict_label_classes: dict,
):
    list_target_label_files = find_txt_files(path_target_label_dir)

    for target_file in tqdm(list_target_label_files):

        target_file_descriptor = "/".join(target_file.split("/")[-3:])

        lines = None
        with open(target_file, "r") as file_data:
            lines = file_data.readlines()

        if len(lines) > 0:
            with open(target_file, "w") as file_data:
                source_labels = dict_label_classes[target_file_descriptor]

                if len(lines) == len(source_labels):
                    for idx_line, line in enumerate(lines):
                        line[0] = str(int(source_labels[idx_line]))

                file_data.writelines(lines)


def map_new_labels_to_label_files(
    path_target_label_dir: str,
    dict_label_classes: dict,
):
    list_target_label_files = find_txt_files(path_target_label_dir)

    for target_file in tqdm(list_target_label_files):

        lines = None
        with open(target_file, "r") as file_data:
            lines = file_data.readlines()

        if len(lines) > 0:
            with open(target_file, "w") as file_data:

                for idx_line, line in enumerate(lines):
                    if len(line) > 0:
                        source_class = int(line[0])
                        target_label = int(dict_label_classes[source_class])
                        lines[idx_line] = str(target_label) + line[1:]

                file_data.writelines(lines)


if __name__ == "__main__":

    dict_source_class_to_target_class_mapping = {
        0: 0,  # source: helmet (0), target: helmet (0)
        1: 2,  # source: no_helmet (1), target: no_helmet (2)
        2: 3,  # source: person (2), target: person (3)
        # 3: , # source: -, target:
    }

    map_new_labels_to_label_files(
        path_target_label_dir="/home/matthias/workspace/Coding/20_helmet_reco/data_hh_3/train/labels",
        dict_label_classes=dict_source_class_to_target_class_mapping,
    )
