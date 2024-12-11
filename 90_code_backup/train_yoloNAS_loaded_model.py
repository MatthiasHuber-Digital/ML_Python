# %%
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
    coco2017_train_yolo_nas,
    coco2017_val_yolo_nas,

)
from super_gradients.training import models
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import (
    DetectionMetrics_050,
    DetectionMetrics_050_095
)
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
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

# Global parameters.
EPOCHS = 100
BATCH_SIZE = 8
WORKERS = 8
super_gradients.setup_device(device='cuda')

train_data = coco_detection_yolo_format_train(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['train_images_dir'],
        'labels_dir': dataset_params['train_labels_dir'],
        'classes': dataset_params['classes'],
        'ignore_empty_annotations': dataset_params['ignore_empty_annotations'],
    },
    dataloader_params={
        'batch_size':BATCH_SIZE,
        'num_workers':WORKERS,
    }
)

val_data = coco_detection_yolo_format_val(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['val_images_dir'],
        'labels_dir': dataset_params['val_labels_dir'],
        'classes': dataset_params['classes'],
        'ignore_empty_annotations': dataset_params['ignore_empty_annotations'],
    },
    dataloader_params={
        'batch_size':BATCH_SIZE,
        'num_workers':WORKERS,
    }
)
print(train_data.dataset.transforms)
#train_data.dataset.plot(plot_transformed_data=True)
train_params = {
    'silent_mode': False,
    "average_best_models": True, # means averaging over all runs
    "warmup_mode": "linear_epoch_step",
    "warmup_initial_lr": 1e-6,
    "lr_warmup_epochs": 3,
    "initial_lr": 1e-4,
    "lr_mode": 'StepLRScheduler',
    "lr_decay_factor": 0.5,
    "lr_updates": [10,20,30,40,50,60,70,80,90,100],
    #"lr_mode": "CosineLRScheduler",
    #"cosine_final_lr_ratio": 0.001,
    "optimizer": "RMSProp", #"Adam",
    "optimizer_params": {"weight_decay": 0.0001},
    "zero_weight_decay_on_bias_and_bn": True,
    "ema": True,
    "ema_params": {"decay": 0.9, "decay_type": "threshold"},
    "max_epochs": EPOCHS,
    "mixed_precision": True, # mixed precision training
    "loss": PPYoloELoss(
        use_static_assigner=False,
        num_classes=len(dataset_params['classes']),
        reg_max=16
    ),
    "finetune": False, # freezes the largest part of the model and leaves only the end tunable.
    "save_ckpt_epoch_list": [5,50,100,150], # give a list of the epochs on which to save the model.
    "save_model": False, # when true, saves ALL best versions of the model, when FALSE only the checkpoint epoch list ones.
    "batch_accumulate": 4, # represents how many batches are run before each backward pass - simulates larger batches
    "load_checkpoint": True, # requires "ckpt_latest.pth" in the "ckpt_root_dir"
    "resume": True,
    "ckpt_root_dir": '/home/matthias/workspace/Coding/00_vista_medizina/10_weights/yoloNAS/pretrained_weights/yoloNAS_s_own_forearm/',
    "valid_metrics_list": [
        DetectionMetrics_050(
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
        """DetectionMetrics_050_095(
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
        )"""
    ],
    #"metric_to_watch": 'mAP@0.50:0.95'
    "metric_to_watch": 'abcdefg'
}

model_yolo_nas_s = torch.load(f='/home/matthias/workspace/Coding/00_vista_medizina/10_weights/yoloNAS/pretrained_weights/ultralytics/yolo_nas_s.pt')
#model_yolo_nas_s = torch.load(f='/home/matthias/workspace/Coding/00_vista_medizina/10_weights/yoloNAS/pretrained_weights/yoloNAS_s_own_forearm/weights_yoloNAS_forearm_map5095_best.pth')
#from ultralytics.models import NAS
#model_yolo_nas_s = NAS.load(f'/home/matthias/workspace/Coding/00_vista_medizina/10_weights/yoloNAS/pretrained_weights/ultralytics/yolo_nas_s.pt')
trainer = Trainer(
        experiment_name='yolo_nas_s',
    )

trainer.train(
    model=model_yolo_nas_s,
    training_params=train_params,
    train_loader=train_data,
    valid_loader=val_data
)