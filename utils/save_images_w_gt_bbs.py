from utils.bounding_box_manipulation import visualize_bounding_boxes


if __name__ == "__main__":
    # path_data_dir = "/home/matthias/workspace/Coding/00_vista_medizina/00_data/2024-11-08/kaggle_bone_fracture_detection_small_2024-11-08/bf_unsharp_masking_bgr_rect"
    # path_data_yaml = Path(path_data_dir) / Path("data.yaml")
    # config = OmegaConf.create(path_data_yaml)
    dict_classes = {
        0: "elbow positive",  # elbow or forearm near elbow. image name was humerus fracture?
        1: "fingers positive",  # fingers -image looked like lower than wrist/forearm
        2: "forearm fracture",  # forearm
        3: "humerus fracture",  # never in GT!
        4: "humerus",  # ( humerus near shoulder, near elbow) - ok
        5: "shoulder fracture",  # shoulder
        6: "wrist positive",  # wrist5
    }
    """
    dict_classes = {
        0: "elbow fracture",
        1: "finger fracture",
        2: "forearm fracture",
        3: "humerus fracture",
        4: "shoulder fracture",
        5: "wrist fracture",
    }
    
    dict_classes = {
        0: "elbow fracture",
        1: "finger fracture",
        2: "forearm fracture",
        3: "humerus fracture",
        4: "shoulder fracture",
        5: "humerus joint fracture",
        6: "bubble",
        7: "wrist fracture",
    }"""
    visualize_bounding_boxes(
        path_images_dir="/home/matthias/workspace/Coding/00_vista_medizina/00_data/2024-11-08/kaggle_bone_fracture_detection_small_2024-11-08/bf_unsharp_masking_bgr_rect/test/images",
        path_gt_labels_dir="/home/matthias/workspace/Coding/00_vista_medizina/00_data/2024-11-08/kaggle_bone_fracture_detection_small_2024-11-08/bf_unsharp_masking_bgr_rect/test/labels",
        path_output_dir="test_images_w_gt_bbs",
        dict_class_names=dict_classes,
        save_images=True,
        show_images=False,
    )
