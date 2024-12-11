import os
from pathlib import Path
import shutil
from tqdm import tqdm

path_base = '/home/matthias/workspace/Coding/00_vista_medizina/00_data/2024-11-08/kaggle_bone_fracture_detection_small_2024-11-08/bf_orig_bgr_poly'

path_images_and_labels = [
    path_base + '/train',
    path_base + '/test',
    path_base + '/val',
]

dict_classes = {
    0: 'elbow_positive', 
    1: 'fingers_positive', 
    2: 'forearm_fracture', 
    3: 'humerus_fracture', 
    4: 'humerus', 
    5: 'shoulder_fracture', 
    6: 'wrist_positive',
}

for path_directory in tqdm(path_images_and_labels):
    print("Processing directory: ", path_directory)

    path_images = path_directory + "/images"
    path_labels = path_directory + "/labels"

    print("Creating class subfolders in the image and label directories...")
    # Generate a folder for each class both in the images directory and the labels directory
    for class_name in list(dict_classes.values()):
        path_class_images = path_images + "/" + class_name
        path_class_labels = path_labels + "/" + class_name

        Path(path_class_images).mkdir(exist_ok=True, parents=True)
        Path(path_class_labels).mkdir(exist_ok=True, parents=True)

    Path(path_images + "/no_fractures").mkdir(exist_ok=True, parents=True)
    Path(path_labels + "/no_fractures").mkdir(exist_ok=True, parents=True)

    del path_class_images, path_class_labels
    
    # List all labels, images and the filenames without extension"
    list_paths_labels = [os.path.join(path_labels, f) for f in os.listdir(path_labels) if f.endswith(".txt")]
    #list_image_filenames_no_ext = [".".join(f.split("/")[-1].split(".")[:-1]) for f in list_paths_labels]
    list_paths_images = [os.path.join(path_images, f) for f in os.listdir(path_images) if f.endswith(".jpg")]

    # For each label, copy the corresponding image into the folder of the class to which it belongs:
    for path_label in tqdm(list_paths_labels):
        current_class = None

        with open(path_label, 'r') as l_txt:
            lines = l_txt.readlines()

            if lines is None or len(lines) == 0:
                path_target_class_labels = path_labels + "/no_fractures"
                path_target_class_images = path_images + "/no_fractures"
                
            else:
                current_class = int(lines[0][0])

                if current_class is not None:
                    path_target_class_images = path_images + "/" + dict_classes[current_class]
                    path_target_class_labels = path_labels + "/" + dict_classes[current_class]

            filename_no_ext = ".".join(path_label.split("/")[-1].split(".")[:-1])

            path_label_destination = path_target_class_labels + "/" + filename_no_ext + ".txt"
            path_image_destination = path_target_class_images + "/" + filename_no_ext + ".jpg"
            path_image_original = path_images + "/" + filename_no_ext + ".jpg"
                    
            try:
                shutil.copy2(path_label, path_label_destination)
            except Exception:
                raise Exception("Error while copying label file: ", filename_no_ext)
            
            try:
                shutil.copy2(path_image_original, path_image_destination)
            except Exception:
                raise Exception("Error while copying image file: ", filename_no_ext)
            """

            if lines is not None and len(lines) > 0:
                current_class = int(lines[0][0])

                if current_class is not None:
                    path_target_class_images = path_images + "/" + dict_classes[current_class]
                    path_target_class_labels = path_labels + "/" + dict_classes[current_class]

                    filename_no_ext = ".".join(path_label.split("/")[-1].split(".")[:-1])

                    path_label_destination = path_target_class_labels + "/" + filename_no_ext + ".txt"
                    path_image_destination = path_target_class_images + "/" + filename_no_ext + ".jpg"
                    path_image_original = path_images + "/" + filename_no_ext + ".jpg"
                    
                    try:
                        shutil.copy2(path_label, path_label_destination)
                    except Exception:
                        raise Exception("Error while copying label file: ", filename_no_ext)
                    
                    try:
                        shutil.copy2(path_image_original, path_image_destination)
                    except Exception:
                        raise Exception("Error while copying image file: ", filename_no_ext)
            """
