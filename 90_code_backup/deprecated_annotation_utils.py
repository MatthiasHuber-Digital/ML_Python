import numpy as np
import os
from PIL import Image
from utils import find_files_of_extension
from tqdm import tqdm


def get_image_size(path_image: str) -> dict:
    # Open the image using PIL's Image class
    with Image.open(path_image) as img:
        # Get the width and height
        return {'width': img.size[0], 'height': img.size[1]}


def convert_polygon_to_yolov7_bbox(
        polygon: list[float], 
    ) -> dict:
    """This function converts polygon annotations to yolov7 rectangular ones.

    Args:
        polygon (list): List of polygon bounding boxes with relative coordinates.

    Returns:
        dict: _description_
    """
    min_x = min(polygon[::2])  # x coordinates are at even indices
    max_x = max(polygon[::2])
    min_y = min(polygon[1::2])  # y coordinates are at odd indices
    max_y = max(polygon[1::2])

    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    width = max_x - min_x
    height = max_y - min_y

    return {'center_x': center_x, 'center_y': center_y, 'width': width, 'height': height}


def convert_single_polygon_annotation_file_to_rectangular(
        path_input_ann_file: str, 
        path_output_ann_file: str, 
    ):
    """Convert polygon annotations in a .txt file to YOLOv7 format and save the .txt.

    Args:
        path_input_ann_file (str): Path of the annotiation input file (polygon format).
        path_output_ann_file (str): Path of the annotation output file (rectangular format).
    """
    data = ''
    with open(path_input_ann_file, 'r') as f_in:
        for line in f_in:
            # Split the line into the polygon coordinates
            polygon_list_split = list(map(float, line.strip().split()))
            class_id = polygon_list_split[0]

            # Convert the polygon to a bounding box
            dict_bb_coords = convert_polygon_to_yolov7_bbox(polygon_list_split[1:]) #, img_width, img_height)
            center_x = dict_bb_coords['center_x']
            center_y = dict_bb_coords['center_y']
            width = dict_bb_coords['width']
            height = dict_bb_coords['height']

            # Write the YOLO format bounding box to the output file
            data += f"{int(class_id)} {center_x} {center_y} {width} {height}\n"
    
    with open(path_output_ann_file, 'w') as f_out:
        f_out.write(data)


def convert_all_polygon_annotations_in_directory_to_rectangular(path_directory: str, replace_files: bool = False):
    list_paths_poly_files = find_files_of_extension(path_root_dir=path_directory, extension="txt")

    print("Converting polygon annotations to rectangular...")
    for path_poly_file in tqdm(list_paths_poly_files):
        if not replace_files:
            poly_path_split = path_poly_file.split("/")
            list_poly_filename = poly_path_split[-1].split(".")[:-1]
            list_poly_filename.append("_rectangular.txt")
            filename_rect = ".".join(list_poly_filename)
            poly_path_split = poly_path_split[:-1]
            poly_path_split.append(filename_rect)
            path_rect_file = "/".join(poly_path_split)
            convert_single_polygon_annotation_file_to_rectangular(path_input_ann_file=path_poly_file, path_output_ann_file=path_rect_file)

        else:
            convert_single_polygon_annotation_file_to_rectangular(path_input_ann_file=path_poly_file, path_output_ann_file=path_poly_file)

    print("--done.")


def load_ground_truth_yolov7(gt_txt_path) -> list[list[float | int]]:
    ground_truths = []
    with open(gt_txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            # Convert from normalized to pixel coordinates
            x_min = float(parts[1]) #* img_width
            y_min = float(parts[2]) #* img_height
            x_max = float(parts[3]) #* img_width
            y_max = float(parts[4]) #* img_height
            
            ground_truths.append([x_min, y_min, x_max, y_max, class_id])

    return ground_truths


def load_ground_truth_polygon(gt_txt_path: str) -> list[list[float | int]]:
    ground_truths = []
    with open(gt_txt_path, 'r') as file:
        # Convert relative (normalized) coordinates to absolute pixel values
        #polygon = np.array([(int(coord * img_width) if i % 2 == 0 else int(coord * img_height)) 
        #                    for i, coord in enumerate(coords)])
        #polygon = polygon.reshape((-1, 1, 2))  # Reshape for cv2.polylines
        for line in file:
            parts = line.strip().split()
            class_id = int(parts[0])
            line_list = [class_id]
            
            idx_part = 1
            while idx_part < len(parts):
                x = float(parts[idx_part])  #* img_width
                y = float(parts[idx_part+1])  #* img_height
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
            x = box.pop(0)  #* img_width
            y = box.pop(0)  #* img_height

            # Calculate top-left and bottom-right corners
            x = int(x * width)
            y = int(y * height)

            coords.append(x)
            coords.append(y)

        box_text = f'GT: {classes[class_id]}'
        
        polygon_points = np.array(coords).reshape((-1, 1, 2))  # Reshape for cv2.polylines
        cv2.polylines(image_data, [polygon_points], isClosed=True, color=(0, 0, 255), thickness=2)
        cv2.putText(image_data, box_text, (coords[0], coords[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


def print_yolov7_format_boxes_to_image(image_data: np.ndarray, boxes_info):

    height, width, __channels = image_data.shape
    
    for box in boxes_info:
        class_id, x_center_frac, y_center_frac, width_frac, height_frac = box

        x_center = x_center_frac * width
        y_center = y_center_frac * height
        width = width_frac * width
        height = height_frac * height

        # Calculate top-left and bottom-right corners
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        box_text = f'GT Class {class_id}'
        
        cv2.rectangle(image_data, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.putText(image_data, box_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


def print_prediction_boxes_to_image(image_data: np.ndarray, boxes_info):
    for idx, box in enumerate(boxes_info.prediction.bboxes_xyxy):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

        class_id = boxes_info.prediction.labels[idx]
        score = boxes_info.prediction.confidence[idx]
        box_text = f'Pred Class {class_id}: {score:.2f}'  # Use actual label or class mapping here
        
        cv2.rectangle(image_data, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(image_data, box_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


def get_label_and_image_paths(path_images_dir: str, path_labels_dir: str) -> tuple[list]:
    list_paths_images = []
    for dirpath, _, filenames in os.walk(path_images_dir):
        
        for filename in filenames:

            if filename.endswith('.' + 'jpg'):
                list_paths_images.append(os.path.join(dirpath, filename))
    list_paths_images

    #list_paths_labels = []
    #for f in list_paths_images:
    #    list_paths_labels.append(".".join(f.split('.')[:-1]) + ".txt")

    list_paths_labels = []
    for dirpath, _, filenames in os.walk(path_labels_dir):
        
        for filename in filenames:

            if filename.endswith('.' + 'txt'):
                list_paths_labels.append(os.path.join(dirpath, filename))
    list_paths_labels

    return {
        'list_paths_images': list_paths_images, 
        'list_paths_labels': list_paths_labels,
    }


def print_polygon_bb_data_as_rectangular_to_image(
    image_data: np.ndarray,
    list_polygons: list[list[float | int]],
):
    height, width, __channels = image_data.shape

    for polygon in list_polygons:
        class_id = int(polygon.pop(0))

        x_min = int(min(polygon[::2]) * width)  # x coordinates are at even indices
        x_max = int(max(polygon[::2]) * width)
        y_min = int(min(polygon[1::2]) * height) # y coordinates are at odd indices
        y_max = int(max(polygon[1::2]) * height)

        box_text = f'GT Class {class_id}'
        
        cv2.rectangle(image_data, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)
        cv2.putText(image_data, box_text, (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


if __name__ == "__main__":
    path_directory = '/home/matthias/workspace/Coding/00_vista_medizina/00_data/2024-11-08/kaggle_bone_fracture_detection_small_2024-11-08/bone_fracture_detection.v4-v4.yolov8_grayscale_rect'
    convert_all_polygon_annotations_in_directory_to_rectangular(path_directory=path_directory, replace_files=True)
