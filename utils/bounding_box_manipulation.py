import cv2
from enum import Enum
import numpy as np
import os
from utils.file_and_path_operations import get_label_and_image_paths
from pathlib import Path
from tqdm import tqdm
from tqdm.auto import tqdm


class OptionsAnnotationsFormat(str, Enum):
    POLYGON_TXT = "Polygon format with txt files. Relative coordinates x1,y1,x2,y2,..."
    RECT_YOLOV7_TXT = "Rectangular yolov7 format with txt files. Relative coordinates center_x, center_y, width_x, height_y"


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
        class_id = polygon[0]

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
    path_gt_labels_dir: str,
):
    dict_paths_labels_images = get_label_and_image_paths(
        path_images_dir=path_images_dir,
        path_gt_labels_dir=path_gt_labels_dir,
    )
    list_paths_images = dict_paths_labels_images["list_paths_images"]

    for path_image in tqdm(list_paths_images):
        image = cv2.imread(path_image)

        path_ann_gt_data = ".".join(path_image.split("/")[-1].split(".")[:-1]) + ".txt"
        path_output_data = path_gt_labels_dir + "/rect_bb/" + path_ann_gt_data
        path_ann_gt_data = path_gt_labels_dir + "/" + path_ann_gt_data

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
    image_data: np.ndarray,
    path_ann_gt_file: str,
    path_ann_gt_new_save_file: str = None,
    save_converted_labels: bool = False,
    dict_class_names: dict = None,
    annotations_format: OptionsAnnotationsFormat = OptionsAnnotationsFormat.RECT_YOLOV7_TXT,
    print_all_annotations_as_rect: bool = True,
):
    """
    Visualize and optionally save bounding boxes from annotation ground truth files.

    This function displays bounding boxes from the annotation ground truth file on the given image.
    It supports various annotation formats and can optionally save converted labels. If the annotation
    file is not found or contains no bounding boxes, it displays this information on the image.

    Args:
        path_ann_gt_file (str): Path to the annotation ground truth file.
        image_data (np.ndarray): Numpy array representing the image data.
        path_ann_gt_new_save_file (str, optional): Path to save the converted annotation file.
            Defaults to None.
        dict_class_names (dict, optional): Dictionary mapping class IDs to class names.
        save_converted_labels (bool, optional): If True, saves the annotations in a converted format.
            Defaults to False.

    """
    if Path(path_ann_gt_file).exists():

        if annotations_format == OptionsAnnotationsFormat.RECT_YOLOV7_TXT:

            ground_truths = load_ground_truth_yolov7(path_ann_gt_file)
            if len(ground_truths) > 0:
                if dict_class_names is not None:
                    for idx, _ in enumerate(ground_truths):
                        ground_truths[idx]["class_id"] = dict_class_names[
                            int(ground_truths[idx]["class_id"])
                        ]
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

        elif annotations_format == OptionsAnnotationsFormat.POLYGON_TXT:

            ground_truths = load_ground_truth_polygon(path_ann_gt_file)
            if dict_class_names is not None:
                for idx, _ in enumerate(ground_truths):
                    ground_truths[idx][0] = dict_class_names[int(ground_truths[idx][0])]
            if print_all_annotations_as_rect:
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


def visualize_bounding_boxes(
    path_images_dir: str,
    path_gt_labels_dir: str,
    path_output_dir: str = None,
    dict_class_names: dict[int, str] = None,
    save_images: bool = True,
    show_images: bool = False,
    model=None,
    iou: float = 0.0,
    conf: float = 0.2,
    max_predictions: int = 2,
):
    """
    Predict and visualize bounding boxes for images, displaying both ground truth and model predictions.

    This function processes images from a specified directory, applies a prediction model (if provided),
    and visualizes both the ground truth and predicted bounding boxes on the images. It can optionally
    save the annotated images and/or display them.

    Args:
        path_images_dir (str): Directory path containing the input images.
        path_gt_labels_dir (str): Directory path containing the ground truth label files.
        path_output_dir (str, optional): Directory path to save the annotated images. Defaults to None.
        dict_class_names (dict, optional):
        save_images (bool, optional): Flag to save the annotated images. Defaults to True.
        show_images (bool, optional): Flag to display the annotated images. Defaults to False.
        model (optional): The prediction model to use. Defaults to None.
        iou (float): Intersection over Union threshold for the predictions. Defaults to 0.0.
        conf (float): Confidence threshold for the predictions. Defaults to 0.2.
        max_predictions (int, optional): Maximum number of predictions to display per image. Defaults to 2.

    Raises:
        ValueError: If both save_images and show_images are False.
        FileNotFoundError: If the specified image or label directories do not exist.
    """
    if not save_images and not show_images:
        raise ValueError("Either save_images or show_images must be True.")

    for path in [path_images_dir, path_gt_labels_dir]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"The directory {path} does not exist.")

    dict_paths_labels_images = get_label_and_image_paths(
        path_images_dir=path_images_dir,
        path_labels_dir=path_gt_labels_dir,
    )
    list_paths_images = dict_paths_labels_images["list_paths_images"]

    print("Processing images...")
    for path_image in tqdm(list_paths_images):

        if os.path.exists(path):
            split_str_path = path_image.split("/")
            filename_no_extension = ".".join(split_str_path[-1].split(".")[:-1])
            path_ann_file_dir = "/".join(split_str_path[:-2]) + "/labels"
            path_ann_gt_file = path_ann_file_dir + "/" + filename_no_extension + ".txt"

            if Path(path_ann_gt_file).exists():
                image_data = cv2.imread(path_image)

                if model is not None:
                    predictions = model.predict(
                        images=path_image,
                        iou=iou,
                        conf=conf,
                        max_predictions=max_predictions,
                    )
                    print_prediction_boxes_to_image(image_data=image_data, boxes_info=predictions)

                # if save_converted_labels:
                #    path_ann_gt_new_save_file = path_ann_file_dir + "/rect_bb/" + filename_no_extension + ".txt"
                # else:
                #    path_ann_gt_file = None

                wrapper_visualize_ann_data(
                    path_ann_gt_file=path_ann_gt_file,
                    image_data=image_data,
                    path_ann_gt_new_save_file=None,
                    dict_class_names=dict_class_names,
                    save_converted_labels=False,
                )

                if show_images:
                    cv2.imshow(path_image.split("/")[-1], image_data)

                if save_images:
                    if path_output_dir is None:
                        raise ValueError(
                            "You wanted to save your images but haven't specified an output directory."
                        )

                    else:
                        path_out_folder = Path(path_output_dir) / Path(split_str_path[-2])
                        if not path_out_folder.exists():
                            Path(path_out_folder).mkdir(exist_ok=False, parents=True)

                        path_image_output_file = path_out_folder / Path(
                            filename_no_extension + ".jpg"
                        )

                        cv2.imwrite(path_image_output_file, image_data)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()

            else:
                print(f"The annotation file {path_ann_gt_file} does not exist. Skipping file.")

        else:
            print(f"The image file {path} does not exist. Skipping file.")
