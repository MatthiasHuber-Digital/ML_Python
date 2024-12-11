
import torch
from super_gradients.training import models
from super_gradients.training import Trainer, models
from super_gradients.training import dataloaders
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train, 
    coco_detection_yolo_format_val,
)
import super_gradients.training
super_gradients.setup_device(device='cuda')
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import (
    DetectionMetrics_050,
    DetectionMetrics,
)
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from tqdm.auto import tqdm

DEVICE = 'cuda:0'
BATCH_SIZE = 1
WORKERS = 8
#model = torch.load('/home/matthias/workspace/Coding/00_vista_medizina/bone_frac_obj_det/yoloNAS/checkpoints/yolo_nas_s/RUN_20241117_212610_007684/ckpt_epoch_100.pth', map_location=torch.device(DEVICE))

ROOT_DIR = '/home/matthias/workspace/Coding/00_vista_medizina/00_data/2024-11-17/forearm_w_aug/fa.v6i.yolov7pytorch'
train_imgs_dir = '/home/matthias/workspace/Coding/00_vista_medizina/00_data/2024-11-17/forearm_w_aug/fa.v6i.yolov7pytorch/train/images'
train_labels_dir = '/home/matthias/workspace/Coding/00_vista_medizina/00_data/2024-11-17/forearm_w_aug/fa.v6i.yolov7pytorch/train/labels'
val_imgs_dir = '/home/matthias/workspace/Coding/00_vista_medizina/00_data/2024-11-17/forearm_w_aug/fa.v6i.yolov7pytorch/valid/images'
val_labels_dir = '/home/matthias/workspace/Coding/00_vista_medizina/00_data/2024-11-17/forearm_w_aug/fa.v6i.yolov7pytorch/valid/labels'
test_imgs_dir = '/home/matthias/workspace/Coding/00_vista_medizina/00_data/2024-11-17/forearm_w_aug/fa.v6i.yolov7pytorch/test/images'
test_labels_dir = '/home/matthias/workspace/Coding/00_vista_medizina/00_data/2024-11-17/forearm_w_aug/fa.v6i.yolov7pytorch/test/labels'
classes = ['forearm_fracture']

dataset_params = {
    'data_dir': ROOT_DIR,
    'train_images_dir': train_imgs_dir,
    'train_labels_dir': train_labels_dir,
    'val_images_dir': val_imgs_dir,
    'val_labels_dir': val_labels_dir,
    'test_images_dir': test_imgs_dir,
    'test_labels_dir': test_labels_dir,
    'classes': classes,
    'ignore_empty_annotations': True,
}

test_data = coco_detection_yolo_format_val(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['test_images_dir'],
        'labels_dir': dataset_params['test_labels_dir'],
        'classes': dataset_params['classes'],
        'ignore_empty_annotations': dataset_params['ignore_empty_annotations'],
    },
    dataloader_params={
        'batch_size':BATCH_SIZE,
        'num_workers':WORKERS,
    }
)

dict_model_paths = {
    'f1@0.5opt': "/home/matthias/workspace/Coding/00_vista_medizina/10_weights/yoloNAS/checkpoints/yolo_nas_s/RUN_20241119_145001_263683/ckpt_best.pth",
    'recall@0.5opt': "/home/matthias/workspace/Coding/00_vista_medizina/10_weights/yoloNAS/checkpoints/yolo_nas_s/RUN_20241119_125916_831963/ckpt_best.pth",
    'mAP50:95@0.5opt': "/home/matthias/workspace/Coding/00_vista_medizina/10_weights/yoloNAS/pretrained_weights/yoloNAS_s_own_forearm/ckpt_best.pth",
}

dict_models_loaded = {}
for model_name in list(dict_model_paths.keys()):
    print("Model: ", model_name)
    dict_models_loaded.update({model_name: 
                        models.get('yolo_nas_s',
                        num_classes=80,
                        checkpoint_path=dict_model_paths[model_name])})
print("Models: ", dict_models_loaded.keys())

trainer = Trainer(
        experiment_name='yolo_nas_s',
    )
"""
for model_name in list(dict_models_loaded.keys()):
    print("Model: ", model_name)
    print(trainer.test(
        model=dict_models_loaded[model_name],
        test_loader=test_data,
        test_metrics_list=DetectionMetrics(
            score_thres=0.5,
            top_k_predictions=300,
            iou_thres=0.25,
            num_cls=len(dataset_params['classes']),
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.1) #0.7
                )))
"""

import os
os.getcwd()

list_paths_files = []
for dirpath, _, filenames in os.walk(dataset_params['test_images_dir']):
    
    for filename in filenames:

        if filename.endswith('.' + 'jpg'):
            list_paths_files.append(os.path.join(dirpath, filename))

list_paths_files

for model in list(dict_models_loaded.keys()):
    print("\n++++++++++++++++++++++++\n+++++++++++++++++++++\nModel: " + model)
    for path_image in list_paths_files:
        prediction = dict_models_loaded[model].predict(path_image)
        #prediction = dict_models_loaded[model](path_image)
        #print(prediction)
        prediction.show()
