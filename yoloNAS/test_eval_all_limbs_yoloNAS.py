import cv2
from enum import Enum
import numpy as np
import os
from pathlib import Path
import torch
from super_gradients.training import models
from super_gradients.training import Trainer, models
from super_gradients.training import dataloaders
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train,
    coco_detection_yolo_format_val,
)
import super_gradients.training

super_gradients.setup_device(device="cuda")
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import (
    DetectionMetrics_050,
    DetectionMetrics,
)
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from tqdm.auto import tqdm
from utils.bounding_box_manipulation import OptionsAnnotationsFormat

DEVICE = "cuda:0"
BATCH_SIZE = 1
WORKERS = 8
TRAINED_FROM_SCRATCH = False


""" class OptionsAnnotationsFormat(str, Enum):
    POLYGON_TXT = "Polygon format with txt files. Relative coordinates x1,y1,x2,y2,..."
    RECT_YOLOV7_TXT = "Rectangular yolov7 format with txt files. Relative coordinates center_x, center_y, width_x, height_y" """


# ROOT_DIR = '/home/matthias/workspace/Coding/00_vista_medizina/00_data/2024-11-08/kaggle_bone_fracture_detection_small_2024-11-08/bf_grayscale_poly'
# ROOT_DIR = '/home/matthias/workspace/Coding/00_vista_medizina/00_data/2024-11-08/kaggle_bone_fracture_detection_small_2024-11-08/bf_orig_bgr_poly' # multi class polygon forearm
ROOT_DIR = "/home/matthias/workspace/Coding/00_vista_medizina/00_data/2024-11-08/kaggle_bone_fracture_detection_small_2024-11-08/bf_grayscale_rect"  # rectangle multi ckass
train_imgs_dir = ROOT_DIR + "/train/images"
train_labels_dir = ROOT_DIR + "/train/labels"
val_imgs_dir = ROOT_DIR + "/val/images"
val_labels_dir = ROOT_DIR + "/val/labels"
test_imgs_dir = ROOT_DIR + "/test/images"
test_labels_dir = ROOT_DIR + "/test/labels"
# classes = ['elbow positive', 'fingers positive', 'forearm fracture', 'humerus fracture', 'humerus', 'shoulder fracture', 'wrist positive']
classes = ["forearm_fracture"]
if TRAINED_FROM_SCRATCH:
    NUM_CLASSES = len(classes)
else:
    NUM_CLASSES = 80  # for the yoloNAS-s architecture
ANNOTATIONS_FORMAT = OptionsAnnotationsFormat.RECT_YOLOV7_TXT  # .RECT_YOLOV7_TXT
PRINT_ALL_ANNOTATIONS_AS_RECT = False
SAVE_POLYGON_ANNOTATIONS_AS_RECT = False

dataset_params = {
    "data_dir": ROOT_DIR,
    "train_images_dir": train_imgs_dir,
    "train_labels_dir": train_labels_dir,
    "val_images_dir": val_imgs_dir,
    "val_labels_dir": val_labels_dir,
    "test_images_dir": test_imgs_dir,
    "test_labels_dir": test_labels_dir,
    "classes": classes,
    "ignore_empty_annotations": True,
}


"""dict_model_paths = {
    #"/home/matthias/workspace/Coding/00_vista_medizina/vista_bone_frac/yoloNAS/checkpoints/yolo_nas_s/RUN_20241123_101156_209514/ckpt_best.pth",
    'f1@0.5opt': '/home/matthias/workspace/Coding/00_vista_medizina/vista_bone_frac/yoloNAS/checkpoints/yolo_nas_s/RUN_20241126_154056_255731/ckpt_epoch_50.pth',
}"""
dict_model_paths = {
    "f1@0.5opt": "/home/matthias/workspace/Coding/00_vista_medizina/10_weights/yoloNAS/checkpoints/yolo_nas_s/f1_0p5_opt/ckpt_best.pth",  # forearm single class model
}


dict_models_loaded = {}
for model_name in list(dict_model_paths.keys()):
    print("Model: ", model_name)
    dict_models_loaded.update(
        {
            model_name: models.get(
                "yolo_nas_s", num_classes=NUM_CLASSES, checkpoint_path=dict_model_paths[model_name]
            )
        }
    )
print("Models: ", dict_models_loaded.keys())


def convert_relative_coords_to_absolute(
    x_y_rel: tuple[float, float],
    img_width_pix: int,
    img_height_px: int,
) -> tuple[int, int]:
    x_y_abs = (
        x_y_rel[0] * img_width_pix,
        x_y_rel[1] * img_height_px,
    )

    return x_y_abs


def load_ground_truth_yolov7(gt_txt_path) -> list[list[float | int]]:
    ground_truths = []
    with open(gt_txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            """class_id = int(parts[0])
            # Convert from normalized to pixel coordinates
            x_min = float(parts[1]) #* img_width
            y_min = float(parts[2]) #* img_height
            x_max = float(parts[3]) #* img_width
            y_max = float(parts[4]) #* img_height
            
            ground_truths.append([x_min, y_min, x_max, y_max, class_id])"""
            class_id = int(parts[0])
            center_x = float(parts[1])  # * img_width
            center_y = float(parts[2])  # * img_height
            width_x = float(parts[3])  # * img_width
            height_y = float(parts[4])  # * img_height

            ground_truths.append(
                {
                    "class_id": class_id,
                    "x_center_frac": center_x,
                    "y_center_frac": center_y,
                    "width_frac": width_x,
                    "height_frac": height_y,
                }
            )

    return ground_truths


def load_ground_truth_polygon(gt_txt_path: str) -> list[list[float | int]]:
    ground_truths = []
    with open(gt_txt_path, "r") as file:
        # Convert relative (normalized) coordinates to absolute pixel values
        # polygon = np.array([(int(coord * img_width) if i % 2 == 0 else int(coord * img_height))
        #                    for i, coord in enumerate(coords)])
        # polygon = polygon.reshape((-1, 1, 2))  # Reshape for cv2.polylines
        for line in file:
            parts = line.strip().split()
            class_id = int(parts[0])
            line_list = [class_id]

            idx_part = 1
            while idx_part < len(parts):
                x = float(parts[idx_part])  # * img_width
                y = float(parts[idx_part + 1])  # * img_height
                line_list.append(x)
                line_list.append(y)

                idx_part += 2  # increment the index to the next x coordinate

            ground_truths.append(line_list)

    return ground_truths


def print_polygon_format_boxes_to_image(image_data: np.ndarray, boxes_info):

    height, width, __channels = image_data.shape

    for box in boxes_info:
        class_id = box.pop(0)
        coords = []
        while len(box) > 0:
            x = box.pop(0)  # * img_width
            y = box.pop(0)  # * img_height

            # Calculate top-left and bottom-right corners
            x = int(x * width)
            y = int(y * height)

            coords.append(x)
            coords.append(y)

        box_text = f"GT: {classes[class_id]}"

        polygon_points = np.array(coords).reshape((-1, 1, 2))  # Reshape for cv2.polylines
        cv2.polylines(image_data, [polygon_points], isClosed=True, color=(0, 0, 255), thickness=2)
        cv2.putText(
            image_data,
            box_text,
            (coords[0], coords[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )


def print_yolov7_format_boxes_to_image(image_data: np.ndarray, boxes_info: dict):

    height, width, __channels = image_data.shape

    for dict_box in boxes_info:
        # conversion from fractional coordinates to pixel coordinates
        x_box_center = dict_box["x_center_frac"] * width
        y_box_center = dict_box["y_center_frac"] * height
        x_box_width = dict_box["width_frac"] * width
        y_box_height = dict_box["height_frac"] * height

        # Calculate top-left and bottom-right corners
        x1 = int(x_box_center - (x_box_width / 2))
        y1 = int(y_box_center - (y_box_height / 2))
        x2 = int(x_box_center + (x_box_width / 2))
        y2 = int(y_box_center + (y_box_height / 2))

        class_id = dict_box["class_id"]
        box_text = f"GT Class {class_id}"

        cv2.rectangle(image_data, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.putText(
            image_data, box_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1
        )


def print_prediction_boxes_to_image(image_data: np.ndarray, boxes_info):
    for idx, box in enumerate(boxes_info.prediction.bboxes_xyxy):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

        class_id = boxes_info.prediction.labels[idx]
        score = boxes_info.prediction.confidence[idx]
        box_text = f"Pred Class {class_id}: {score:.2f}"  # Use actual label or class mapping here

        cv2.rectangle(image_data, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(
            image_data, box_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1
        )


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


def print_polygon_bb_data_as_rectangular_to_image(
    image_data: np.ndarray,
    list_polygons: list[list[float | int]],
):
    """This function prints polygon bounding box data (list, class-id and point relative coords) to the image).

    Args:
        image_data (np.ndarray): The image data in numpy format.
        list_polygons (list[list[float  |  int]]): Each list has as entry 0 the class-id.
        The other elements in each list are the x (even indices) and y coordinates (odd indices) of the points.
    """
    height, width, __channels = image_data.shape

    for polygon in list_polygons:
        class_id = int(polygon[0])

        dict_rectangle_corners = convert_polygon_points_to_rectangle_corner_points(polygon[1:])
        x1 = int(dict_rectangle_corners["x1"] * width)  # x coordinates are at even indices
        x2 = int(dict_rectangle_corners["x2"] * width)
        y1 = int(dict_rectangle_corners["y1"] * height)  # y coordinates are at odd indices
        y2 = int(dict_rectangle_corners["y2"] * height)

        box_text = f"Ground truth: {class_id}"

        cv2.rectangle(image_data, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.putText(
            image_data, box_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1
        )


def convert_polygon_points_to_rectangle_corner_points(
    list_data_points: list[float | int],
) -> dict[str, float | int]:
    """This function converts an arbitrary length list of polygon points to a 2-tuple of rectangle corner points.

    The algorithm works for both relative and absolute coordinates.

    Args:
        list_data_points (list[float | int]): List of polygon points.

    Returns:
        dict[str,float|int]: Dictionary of rectangle corner points.
    """
    x1 = min(list_data_points[::2])  # x coordinates are at even indices
    x2 = max(list_data_points[::2])

    y1 = min(list_data_points[1::2])  # y coordinates are at odd indices
    y2 = max(list_data_points[1::2])

    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


# Function to convert bounding box to YOLO format
def convert_rectangle_corner_points_to_yolo_format(
    dict_rect_corner_pts: dict[float | int],
    image_data: np.ndarray = None,
    input_abs_coords: bool = True,
) -> dict[float | int]:
    """This function converts rectangle bounding box corner points to yolo format.

    The algorithm works for both relative and absolute coordinates.

    Absolute coordinates require normaliation by the image width and height, which is why input_abs_coords=True is required in that case.

    Args:
        dict_rect_corner_pts (dict[float | int]): _description_
        image_data (np.ndarray): _description_

    Returns:
        dict[float|int]: _description_
    """

    # Calculate center of the box (x_center, y_center)
    box_center_x = (dict_rect_corner_pts["x1"] + dict_rect_corner_pts["x2"]) / 2
    box_center_y = (dict_rect_corner_pts["y1"] + dict_rect_corner_pts["y2"]) / 2

    # Calculate width and height of the bounding box
    box_width_x = dict_rect_corner_pts["x2"] - dict_rect_corner_pts["x1"]
    box_height_y = dict_rect_corner_pts["y2"] - dict_rect_corner_pts["y1"]

    # Normalize the values by the width and height of the image
    if input_abs_coords:
        if image_data is not None:
            image_height, image_width, __channels = image_data.shape
            box_center_x /= image_width
            box_center_y /= image_height
            box_width_x /= image_width
            box_height_y /= image_height
        else:
            raise ValueError("Image data is required for absolute coordinate conversion.")

    return {
        "box_center_x": box_center_x,
        "box_center_y": box_center_y,
        "box_width_x": box_width_x,
        "box_height_y": box_height_y,
    }


def save_polygon_bbs_to_yolo_annotations(
    image_data: np.ndarray,
    list_boxes: list[list],
    path_label_output: str,
):
    path_parent_dir = "/".join(path_label_output.split("/")[:-1])
    if not Path(path_parent_dir).exists():
        Path(path_parent_dir).mkdir(exist_ok=False, parents=True)

    with open(path_label_output, "w") as f:
        for list_bbox in list_boxes:
            if len(list_bbox) > 0:
                class_id = list_bbox[0]

                dict_rectangle_points = convert_polygon_points_to_rectangle_corner_points(
                    list_data_points=list_bbox[1:],
                )
                dict_yolo_rect_coords = convert_rectangle_corner_points_to_yolo_format(
                    dict_rect_corner_pts=dict_rectangle_points,
                    image_data=image_data,
                    input_abs_coords=False,
                )

                x_center, y_center = (
                    dict_yolo_rect_coords["box_center_x"],
                    dict_yolo_rect_coords["box_center_y"],
                )
                width, height = (
                    dict_yolo_rect_coords["box_width_x"],
                    dict_yolo_rect_coords["box_height_y"],
                )

                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


def save_dataset_polygon_annotations_as_rectangle(
    path_images_dir: str,
    path_labels_dir: str,
):
    dict_paths_labels_images = get_label_and_image_paths(
        path_images_dir=path_images_dir,
        path_labels_dir=path_labels_dir,
    )
    list_paths_images = dict_paths_labels_images["list_paths_images"]

    for path_image in tqdm(list_paths_images):
        image = cv2.imread(path_image)

        path_ann_gt_data = ".".join(path_image.split("/")[-1].split(".")[:-1]) + ".txt"
        path_output_data = path_labels_dir + "/rect_bb/" + path_ann_gt_data
        path_ann_gt_data = path_labels_dir + "/" + path_ann_gt_data

        if Path(path_ann_gt_data).exists():

            if ANNOTATIONS_FORMAT == OptionsAnnotationsFormat.POLYGON_TXT:

                ground_truths = load_ground_truth_polygon(path_ann_gt_data)
                save_polygon_bbs_to_yolo_annotations(
                    image_data=image, list_boxes=ground_truths, path_label_output=path_output_data
                )

            else:
                raise TypeError(
                    "The file {path_ann_gt_data} is NOT a polygon format annotations file."
                )


def wrapper_visualize_ann_data(
    path_ann_gt_file: str,
    image_data: np.ndarray,
    path_ann_gt_new_save_file: str = None,
    save_converted_labels: bool = False,
):
    """This function displays bounding boxes from the annotation ground truth file.

    The function can display various formats. In case the file is not found, or in case there is no bounding box
    in the picture, this will be displayed as information in the picture at its top.

    Args:
        path_ann_gt_file (str): Path to the annotation ground truth file.
        image_data (np.ndarray): Numpy image data.
        path_ann_gt_new_save_file (str, optional): Path to the annotation file to which the converted data
            needs to be saved. Defaults to None.
        save_converted_labels (bool, optional): True, if the annotation should be saved in another format. Defaults to False.
    """
    if Path(path_ann_gt_file).exists():

        if ANNOTATIONS_FORMAT == OptionsAnnotationsFormat.RECT_YOLOV7_TXT:

            ground_truths = load_ground_truth_yolov7(path_ann_gt_file)
            if len(ground_truths) > 0:
                print_yolov7_format_boxes_to_image(image_data=image_data, boxes_info=ground_truths)
            else:
                cv2.putText(
                    image_data,
                    "Ground truth: no fractures",
                    (0, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 255),
                    1,
                )

        elif ANNOTATIONS_FORMAT == OptionsAnnotationsFormat.POLYGON_TXT:

            ground_truths = load_ground_truth_polygon(path_ann_gt_file)

            if PRINT_ALL_ANNOTATIONS_AS_RECT:
                print_polygon_bb_data_as_rectangular_to_image(
                    image_data=image_data, list_polygons=ground_truths
                )
                if save_converted_labels:
                    save_polygon_bbs_to_yolo_annotations(
                        image_data=image_data,
                        list_boxes=ground_truths,
                        path_label_output=path_ann_gt_new_save_file,
                    )
            else:
                print_polygon_format_boxes_to_image(image_data=image_data, boxes_info=ground_truths)

    else:

        cv2.putText(
            image_data,
            "Ground truth: no fractures",
            (0, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
        )


def predict_and_visualize_bounding_boxes(
    path_images_dir: str,
    path_labels_dir: str,
    save_converted_labels: bool = False,
):

    for path in [path_images_dir, path_labels_dir]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"The directory {path} does not exist.")

    dict_paths_labels_images = get_label_and_image_paths(
        path_images_dir=path_images_dir,
        path_labels_dir=path_labels_dir,
    )
    list_paths_images = dict_paths_labels_images["list_paths_images"]

    for path_image in list_paths_images:

        image_data = cv2.imread(path_image)
        predictions = dict_models_loaded[list(dict_models_loaded.keys())[0]].predict(
            images=path_image,
            iou=0.0,
            conf=0.2,
            max_predictions=2,
        )

        split_str_path = path_image.split("/")

        filename_no_extension = ".".join(split_str_path[-1].split(".")[:-1])
        path_ann_file_dir = "/".join(split_str_path[:-2]) + "/labels"

        path_ann_gt_file = path_ann_file_dir + "/" + filename_no_extension + ".txt"
        path_ann_gt_new_save_file = path_ann_file_dir + "/rect_bb/" + filename_no_extension + ".txt"

        wrapper_visualize_ann_data(
            path_ann_gt_file=path_ann_gt_file,
            image_data=image_data,
            path_ann_gt_new_save_file=path_ann_gt_new_save_file,
            save_converted_labels=save_converted_labels,
        )

        print_prediction_boxes_to_image(image_data=image_data, boxes_info=predictions)

        print("Processing image: " + path_image.split("/")[-1])
        cv2.imshow(path_image.split("/")[-1], image_data)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # predict_and_visualize_bounding_boxes(path_images_dir=dataset_params['test_images_dir'], path_labels_dir=dataset_params['test_labels_dir'], save_converted_labels=SAVE_POLYGON_ANNOTATIONS_AS_RECT)
    predict_and_visualize_bounding_boxes(
        path_images_dir=dataset_params["train_images_dir"],
        path_labels_dir=dataset_params["train_labels_dir"],
        save_converted_labels=SAVE_POLYGON_ANNOTATIONS_AS_RECT,
    )
    # save_dataset_polygon_annotations_as_rectangle(path_images_dir=dataset_params['test_images_dir'], path_labels_dir=dataset_params['test_labels_dir'])
    # save_dataset_polygon_annotations_as_rectangle(path_images_dir=dataset_params['train_images_dir'], path_labels_dir=dataset_params['train_labels_dir'])
    # save_dataset_polygon_annotations_as_rectangle(path_images_dir=dataset_params['val_images_dir'], path_labels_dir=dataset_params['val_labels_dir'])


"""
    def predict(
        self,
        images: ImageSource,
        iou: Optional[float] = None,
        conf: Optional[float] = None,
        batch_size: int = 32,
        fuse_model: bool = True,
        skip_image_resizing: bool = False,
        nms_top_k: Optional[int] = None,
        max_predictions: Optional[int] = None,
        multi_label_per_box: Optional[bool] = None,
        class_agnostic_nms: Optional[bool] = None,
        fp16: bool = True,
    ) -> ImagesDetectionPrediction:
        Predict an image or a list of images.

        :param images:              Images to predict.
        :param iou:                 (Optional) IoU threshold for the nms algorithm. If None, the default value associated to the training is used.
        :param conf:                (Optional) Below the confidence threshold, prediction are discarded.
                                    If None, the default value associated to the training is used.
        :param batch_size:          Maximum number of images to process at the same time.
        :param fuse_model:          If True, create a copy of the model, and fuse some of its layers to increase performance. This increases memory usage.
        :param skip_image_resizing: If True, the image processor will not resize the images.
        :param nms_top_k:           (Optional) The maximum number of detections to consider for NMS.
        :param max_predictions:     (Optional) The maximum number of detections to return.
        :param multi_label_per_box: (Optional) If True, each anchor can produce multiple labels of different classes.
                                    If False, each anchor can produce only one label of the class with the highest score.
        :param class_agnostic_nms:  (Optional) If True, perform class-agnostic NMS (i.e IoU of boxes of different classes is checked).
                                    If False NMS is performed separately for each class.
        :param fp16:                        If True, use mixed precision for inference.
"""
