from collections import OrderedDict
import cv2
from enum import Enum
from efficientnet_pytorch import EfficientNet
import glob
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import os
from pathlib import Path
import PIL.Image
import torch
from torchvision.models import *
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize, Compose, ToPILImage
from torchvision import datasets

from utils.cnn_grad_visualization.visualisation.core.utils import *
from utils.cnn_grad_visualization.visualisation.core.utils import device
from utils.cnn_grad_visualization.visualisation.core.utils import image_net_postprocessing
from utils.cnn_grad_visualization.visualisation.core import *
from utils.cnn_grad_visualization.visualisation.core.utils import image_net_preprocessing
from efficient_net.utils import load_efficient_net_model
from efficient_net.options import OptionsModelType


def tensor2img(tensor, ax=plt):
    tensor = tensor.squeeze()
    if len(tensor.shape) > 2:
        tensor = tensor.permute(1, 2, 0)
    img = tensor.detach().cpu().numpy()
    return img


def save_image_CAM_plot(
    image: np.ndarray,
    image_filename: str,
    predicted_CAM: np.ndarray,
    gt_label: str,
    pred_label: str,
    conf_score: float,
    output_dir: str = "output_images",
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a figure with 2 subplots (original image and predicted image)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))

    # Plot original image
    ax1.imshow(image)
    ax1.set_title(
        f"{image_filename}\nGround truth: {gt_label}\nPred: {pred_label}\nPred. confidence: {conf_score}"
    )
    ax1.axis("off")  # Hide axes

    # Plot predicted image
    ax2.imshow(predicted_CAM)
    ax2.set_title("Prediction class activation map")
    ax2.axis("off")  # Hide axes

    # Save the figure with both images
    path_save_file = f"{output_dir}/{image_filename}.png"
    plt.tight_layout()
    plt.savefig(path_save_file, dpi=300)  # Save each frame as a PNG file
    plt.close(fig)  # Close the figure to avoid memory overload


def save_images_as_gif(images: list[np.ndarray], predicted_CAMs: list[np.ndarray]):

    fig, ax1 = plt.subplots(1, 2, figsize=(20, 20))

    def update(frame):
        all_ax = []
        ax1.set_yticklabels([])
        ax1.set_xticklabels([])
        ax1.text(1, 1, "Orig. Im", color="white", ha="left", va="top", fontsize=30)
        all_ax.append(ax1.imshow(images[frame]))

        ax1.set_yticklabels([])
        ax1.set_xticklabels([])
        ax1.text(1, 1, "efficientnet-b4", color="white", ha="left", va="top", fontsize=20)
        ax1.imshow(predicted_CAMs[frame], animated=True)

        return all_ax

    ani = FuncAnimation(fig, update, frames=range(len(images)), interval=5000, blit=True)
    fig.tight_layout()
    ani.save("class_activation_maps.gif", writer="imagemagick")


max_img = 10000
model_type = OptionsModelType.B4


class OptionsSaveFormat(str, Enum):
    GIF = "Save multiple images as GIF"
    JPG = "Save each image as JPG"


save_format = OptionsSaveFormat.JPG
max_plotted_images = None
path_test_dataset = "/home/matthias/workspace/Coding/00_vista_medizina/00_data/2024-11-08/kaggle_bone_fracture_detection_small_2024-11-08/bf_orig_bgr_classification/test"

images = []
all_paths = glob.glob(f"{path_test_dataset}/**/*")

image_paths = [x for x in all_paths if Path(x).is_file()]
# some_images_loaded = list(map(lambda x: PIL.Image.open(x), image_paths[:max_img]))
# images.extend(some_images_loaded)

""" inputs = [
    Compose([Resize((224, 224)), ToTensor(), image_net_preprocessing])(x).unsqueeze(0)
    for x in images
]  # add 1 dim for batch """
# inputs = [i.to(device) for i in inputs]
test_transforms = transforms.Compose(
    [
        # Transforms-Resize by default keeps the aspect ratio in case you hand over an INT
        transforms.Resize(model_type.get_resolution()),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
show_transforms = transforms.Compose(
    [
        transforms.Resize(model_type.get_resolution()),
    ]
)
# Transforms-Resize by default keeps the aspect ratio in case you hand over an INT
# images = list(map(lambda x: cv2.resize(np.array(x), (224, 224)), images))  # resize i/p img
# transforms.ToTensor(),
preprocessed_inputs = [test_transforms(np.array(x)).unsqueeze(0).to(device) for x in images]
images = [show_transforms(np.array(x)) for x in images]

model = load_efficient_net_model(
    path_model="/home/matthias/workspace/Coding/00_vista_medizina/10_weights/efficient_net/2024-12-02_bf_kaggle/efficientnet-b4_BEST.pth",
    model_type=model_type,
    num_classes=7,
)
test_dataset = datasets.ImageFolder(path_test_dataset, transform=None)
list_classes = test_dataset.classes
model = model.to(device)
model.cuda().eval()
# model.eval()

vis = GradCam(model, device)

if max_plotted_images is not None:
    image_paths = image_paths[:max_plotted_images]

for path_image in image_paths:
    split_path_str = path_image.split("/")
    gt_class_label = split_path_str[-2]
    filename = split_path_str[-1]

    original_image = PIL.Image.open(path_image)
    # preprocessed_image = test_transforms(np.array(original_image)).unsqueeze(0).to(device)
    preprocessed_image = test_transforms(original_image).unsqueeze(0).to(device)

    CAM_tensor, dict_prediction = vis(
        preprocessed_image, model._conv_head, postprocessing=image_net_postprocessing
    )
    predicted_class = dict_prediction["predicted_class"]
    predicted_CAM_image = tensor2img(CAM_tensor)
    """predicted_CAM = tensor2img(
        vis(preprocessed_image, model._conv_head, postprocessing=image_net_postprocessing)[0]
    )"""
    original_image = show_transforms(original_image)

    if save_format == OptionsSaveFormat.JPG:

        save_image_CAM_plot(
            image=original_image,
            image_filename=filename,
            predicted_CAM=predicted_CAM_image,
            gt_label=gt_class_label,
            pred_label=list_classes[dict_prediction["predicted_class"]],
            conf_score=dict_prediction["confidence"],
            output_dir="output_images/" + gt_class_label,
        )
"""
predicted_CAMs = list(
    # map(lambda x: tensor2img(vis(x, None, postprocessing=image_net_postprocessing)[0]), inputs)
    map(
        lambda x: tensor2img(vis(x, model._conv_head, postprocessing=image_net_postprocessing)[0]),
        inputs,
    )
)
torch.cuda.empty_cache()


class OptionsSaveFormat(str, Enum):
    GIF = "Save multiple images as GIF"
    JPG = "Save each image as JPG"


save_format = OptionsSaveFormat.JPG

if save_format == OptionsSaveFormat.GIF:

    fig, ax1 = plt.subplots(1, 2, figsize=(20, 20))

    def update(frame):
        all_ax = []
        ax1.set_yticklabels([])
        ax1.set_xticklabels([])
        ax1.text(1, 1, "Orig. Im", color="white", ha="left", va="top", fontsize=30)
        all_ax.append(ax1.imshow(images[frame]))

        ax1.set_yticklabels([])
        ax1.set_xticklabels([])
        ax1.text(1, 1, "efficientnet-b4", color="white", ha="left", va="top", fontsize=20)
        ax1.imshow(predicted_CAMs[frame], animated=True)

        return all_ax

    ani = FuncAnimation(fig, update, frames=range(len(images)), interval=5000, blit=True)
    fig.tight_layout()
    ani.save("class_activation_maps.gif", writer="imagemagick")

elif save_format == OptionsSaveFormat.JPG:

    # Assuming images and predicted_CAMs are already defined as lists or arrays

    def save_images(images, predicted_CAMs, output_dir="output_images"):
        # Make sure output directory exists (you can use os.makedirs to create it)
        import os

        if not os.path_test_dataset.exists(output_dir):
            os.makedirs(output_dir)

        for image in range(len(images)):
            # Create a figure with 2 subplots (original image and predicted image)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))

            # Plot original image
            ax1.imshow(images[image])
            ax1.set_title("Original Image")
            ax1.axis("off")  # Hide axes

            # Plot predicted image
            ax2.imshow(predicted_CAMs[image])
            ax2.set_title("Prediction")
            ax2.axis("off")  # Hide axes

            # Save the figure with both images
            file_name = f"{output_dir}/frame_{image:04d}.png"
            plt.tight_layout()
            plt.savefig(file_name, dpi=300)  # Save each frame as a PNG file
            plt.close(fig)  # Close the figure to avoid memory overload

    save_images(images, predicted_CAMs, output_dir="test_dataset_class_activation_maps")
"""
