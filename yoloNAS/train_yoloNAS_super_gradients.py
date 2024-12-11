"""
USAGE INSTRUCTIONS:

I. Set the necesary dataset and training parameters.
- the coco_detection_yolo_format_train/val dataloaders can read data in the yolov7 .txt format: 
<class-id(int)> <x1(float, relative coords)> <y1(float, relative coords)> <x2(float, relative coords)> <y2(float, relative coords)>

II.1) For loading an untrained model architecture (currently this is the only variant working with multiclass) - 
set TRAIN_FROM_SCRATCH = True
(number of classes =len(classes) in the loss and detection metrics specification in the training parameters dict)
II.2) For loading a trained model architecture (currently the only architecture working with single class):
set TRAIN_FROM_SCRATCH = False
(number of classes has to be 80 for the loss and the detection metric specification in the training parameters dict)

III. Choose one or several detection metrics. See commented section.

"""

from super_gradients.common.object_names import Models
from super_gradients.training import Trainer, models
from super_gradients.training.metrics.classification_metrics import Accuracy, Top5
from super_gradients.training.dataloaders.dataloaders import cifar10_train, cifar10_val
import torch

# https://learnopencv.com/train-yolo-nas-on-custom-dataset/
import super_gradients
from super_gradients.training import Trainer
from super_gradients.training import dataloaders
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train,
    coco_detection_yolo_format_val,
)
from super_gradients.training import models
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import (
    DetectionMetrics_050,
    DetectionMetrics_050_095,
)
from super_gradients.training.models.detection_models.pp_yolo_e import (
    PPYoloEPostPredictionCallback,
)
from tqdm.auto import tqdm

import os
import requests
import zipfile
import cv2
import matplotlib.pyplot as plt
import glob
import numpy as np
import random

from ultralytics import NAS

# Global parameters.
EPOCHS = 50
BATCH_SIZE = 8
WORKERS = 8
super_gradients.setup_device(device="cuda")
TRAIN_FROM_SCRATCH = False

ROOT_DIR = "/home/matthias/workspace/Coding/00_vista_medizina/00_data/2024-11-08/kaggle_bone_fracture_detection_small_2024-11-08/bf_histo_bgr_rect"
train_imgs_dir = ROOT_DIR + "/train/images"
train_labels_dir = ROOT_DIR + "/train/labels"
val_imgs_dir = ROOT_DIR + "/val/images"
val_labels_dir = ROOT_DIR + "/val/labels"
test_imgs_dir = ROOT_DIR + "/test/images"
test_labels_dir = ROOT_DIR + "/test/labels"
classes = [
    "elbow positive",
    "fingers positive",
    "forearm fracture",
    "humerus fracture",
    "humerus",
    "shoulder fracture",
    "wrist positive",
]

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

if TRAIN_FROM_SCRATCH:
    NUM_CLASSES = len(classes)
    model_yolo_nas_s = models.get("yolo_nas_s", num_classes=len(dataset_params["classes"]))
else:
    NUM_CLASSES = 80  # for the yoloNAS-s architecture
    # TRAINING FROM PRETRAINED ULTRALYTICS MODEL for yoloNAS-s:
    model_yolo_nas_s = torch.load(
        f="/home/matthias/workspace/Coding/00_vista_medizina/10_weights/yoloNAS/pretrained_weights/ultralytics/yolo_nas_s.pt"
    )

trainer = Trainer(
    experiment_name="yolo_nas_s",
)

train_data = coco_detection_yolo_format_train(
    dataset_params={
        "data_dir": dataset_params["data_dir"],
        "images_dir": dataset_params["train_images_dir"],
        "labels_dir": dataset_params["train_labels_dir"],
        "classes": dataset_params["classes"],
        "cache_annotations": False,  # if FALSE, then the annotations are always read-in each time a training is being triggered.
        "ignore_empty_annotations": dataset_params["ignore_empty_annotations"],
    },
    dataloader_params={
        "batch_size": BATCH_SIZE,
        "num_workers": WORKERS,
    },
)

val_data = coco_detection_yolo_format_val(
    dataset_params={
        "data_dir": dataset_params["data_dir"],
        "images_dir": dataset_params["val_images_dir"],
        "labels_dir": dataset_params["val_labels_dir"],
        "classes": dataset_params["classes"],
        "ignore_empty_annotations": dataset_params["ignore_empty_annotations"],
    },
    dataloader_params={
        "batch_size": BATCH_SIZE,
        "num_workers": WORKERS,
    },
)
print(train_data.dataset.transforms)
# train_data.dataset.plot(plot_transformed_data=True)
train_params = {
    "silent_mode": False,
    "average_best_models": True,  # means averaging over all runs
    "warmup_mode": "linear_epoch_step",
    "warmup_initial_lr": 1e-6,
    "lr_warmup_epochs": 3,
    "initial_lr": 1e-4,
    "lr_mode": "StepLRScheduler",
    "lr_decay_factor": 0.5,
    "lr_updates": [10, 20, 30, 40, 50],  # , 60, 70, 80, 90, 100],
    # "lr_mode": "CosineLRScheduler",
    # "cosine_final_lr_ratio": 0.001,
    "optimizer": "RMSProp",  # "Adam",
    "optimizer_params": {"weight_decay": 0.0001},
    "zero_weight_decay_on_bias_and_bn": True,
    "ema": True,
    "ema_params": {"decay": 0.9, "decay_type": "threshold"},
    "max_epochs": EPOCHS,
    "mixed_precision": True,  # mixed precision training
    "loss": PPYoloELoss(use_static_assigner=False, num_classes=NUM_CLASSES, reg_max=16),
    "finetune": False,  # freezes the largest part of the model and leaves only the end tunable.
    "save_ckpt_epoch_list": [
        20,
        40,
    ],  # [20,50, 75, 100], # give a list of the epochs on which to save the model.
    "save_model": False,  # when true, saves ALL best versions of the model, when FALSE only the checkpoint epoch list ones.
    "batch_accumulate": 4,  # represents how many batches are run before each backward pass - simulates larger batches
    # "load_checkpoint": True, # requires "ckpt_latest.pth" in the "ckpt_root_dir"
    # "resume": True,
    # "ckpt_root_dir": '/home/matthias/workspace/Coding/00_vista_medizina/10_weights/yoloNAS/pretrained_weights/yoloNAS_s_own_forearm/',
    "valid_metrics_list": [
        DetectionMetrics_050(
            score_thres=0.5,
            top_k_predictions=300,
            num_cls=NUM_CLASSES,
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.5,
                nms_top_k=300,
                max_predictions=300,
                nms_threshold=0.7,
            ),
        ),
    ],
    "metric_to_watch": "F1@0.50",
}
"""        DetectionMetrics_050_095(
            score_thres=0.1,
            top_k_predictions=300,
            num_cls=len(dataset_params['classes']),
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.7
            )
        ),
        "metrics_to_watch": ....
        Available metrics to monitor are: `['PPYoloELoss/loss_cls', 'PPYoloELoss/loss_iou', 
    'PPYoloELoss/loss_dfl', 'PPYoloELoss/loss', 'Precision@0.50', 'Recall@0.50', 'mAP@0.50', 
    'F1@0.50', 'Best_score_threshold']"""

# TRAINING FROM PRETRAINED ULTRALYTICS MODEL:
# model_yolo_nas_s = torch.load(f='/home/matthias/workspace/Coding/00_vista_medizina/10_weights/yoloNAS/pretrained_weights/ultralytics/yolo_nas_s.pt')
# model_yolo_nas_s = torch.load(f='/home/matthias/workspace/Coding/00_vista_medizina/10_weights/yoloNAS/pretrained_weights/yoloNAS_s_own_forearm/weights_yoloNAS_forearm_map5095_best.pth')
# from ultralytics.models import NAS
# model_yolo_nas_s = NAS.load(f'/home/matthias/workspace/Coding/00_vista_medizina/10_weights/yoloNAS/pretrained_weights/ultralytics/yolo_nas_s.pt')

trainer.train(
    model=model_yolo_nas_s,
    training_params=train_params,
    train_loader=train_data,
    valid_loader=val_data,
)
