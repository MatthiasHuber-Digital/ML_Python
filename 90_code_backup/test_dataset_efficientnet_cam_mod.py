import cv2
from efficientnet_pytorch import EfficientNet
import matplotlib.pyplot as plt
import numpy as np

# from omegaconf.omegaconf import OmegaConf
import os
import shutil
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    auc,
)
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.transforms import Compose, Normalize
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from efficient_net.options import OptionsModelType
from efficient_net.utils import load_efficient_net_model
from utils.cnn_grad_visualization.visualisation.core import GradCam
from utils.cnn_grad_visualization.visualization.core.utils import *
from utils.cnn_grad_visualization.utils import tensor2img, subplot


image_net_mean = torch.Tensor([0.485, 0.456, 0.406])
image_net_std = torch.Tensor([0.229, 0.224, 0.225])


class NormalizeInverse(Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.Tensor(mean)
        std = torch.Tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


image_net_preprocessing = Compose([Normalize(mean=image_net_mean, std=image_net_std)])

image_net_postprocessing = Compose([NormalizeInverse(mean=image_net_mean, std=image_net_std)])


def tensor2cam(image, cam):
    image_with_heatmap = image2cam(
        image.squeeze().permute(1, 2, 0).cpu().numpy(), cam.detach().cpu().numpy()
    )

    return torch.from_numpy(image_with_heatmap).permute(2, 0, 1)


def image2cam(image, cam):
    h, w, c = image.shape
    cam -= np.min(cam)
    cam /= np.max(cam)  # Normalize between 0-1
    cam = cv2.resize(cam, (w, h))

    cam = np.uint8(cam * 255.0)
    img_with_cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    img_with_cam = cv2.cvtColor(img_with_cam, cv2.COLOR_BGR2RGB)
    img_with_cam = img_with_cam + (image * 255)
    img_with_cam /= np.max(img_with_cam)

    return img_with_cam


def convert_to_grayscale(cv2im):
    """
        Converts 3d image to grayscale

    Args:
        cv2im (numpy arr): RGB image with shape (D,W,H)

    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)

    credits to https://github.com/utkuozbulak/pytorch-cnn-visualizations
    """
    grayscale_im = np.sum(np.abs(cv2im), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1)
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im


def imshow(tensor):
    tensor = tensor.squeeze()
    if len(tensor.shape) > 2:
        tensor = tensor.permute(1, 2, 0)
    img = tensor.cpu().numpy()
    plt.imshow(img, cmap="gray")
    plt.show()


def gradcam2crop(
    cam_object: GradCam,
    original_img: np.ndarray,
    preprocessed_img: np.ndarray,
    TR: 0.3,
) -> np.ndarray:
    _bs, _c, w, h = preprocessed_img.shape
    cam_object = cam_object.numpy()
    cam_object -= np.min(cam_object)
    cam_object /= np.max(cam_object)

    cam_object = cv2.resize(cam_object, (w, h))
    mask = cam_object > TR

    original_img = tensor2img(image_net_postprocessing(original_img[0].squeeze()))

    crop = original_img.copy()
    crop[mask == 0] = 0

    return crop


def vis_outs2images_classes(list_predictions: list, list_classes: list) -> tuple:
    images = [x[0] for x in predictions]
    classes = [list_classes[x] for x in predictions]

    return images, classes


def get_args(
    model_type: OptionsModelType,
    path_test_data_dir: str,
    path_trained_model: str,
    num_classes: int,
):
    dict_args = {
        "model_type": model_type,
        "path_test_data_dir": path_test_data_dir,
        "path_trained_model": path_trained_model,
        "num_classes": num_classes,
        "path_output_dir": "predictions",
        "batch_size": 1,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "print_class_activation_maps": True,
    }

    # args = OmegaConf.create(dict_args)

    return dict_args


def test(opt):
    # Load the model
    model = load_efficient_net_model(
        path_model=opt["path_trained_model"],
        model_type=opt["model_type"],
        num_classes=opt["num_classes"],
    )

    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cuda"):
        model.cuda().eval()
    else:
        model.eval().eval()

    print_class_activation_maps = opt["print_class_activation_maps"]

    # Data transformations
    test_transforms = transforms.Compose(
        [
            transforms.Resize(opt["model_type"].get_resolution()),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    show_transforms = transforms.Compose(
        [
            transforms.Resize(opt["model_type"].get_resolution()),
            transforms.ToTensor(),
        ]
    )

    # Create the output directory
    print(
        "Create the output directory (or overwrite the existing one): %s" % opt["path_output_dir"]
    )
    if os.path.isdir(opt["path_output_dir"]):
        shutil.rmtree(opt["path_output_dir"])
    os.makedirs(opt["path_output_dir"])

    # Initialize TensorBoard writer
    writer = SummaryWriter(opt["path_output_dir"])

    # Load dataset and dataloader
    print("Load the datasets")
    test_dataset = datasets.ImageFolder(opt["path_test_data_dir"], transform=test_transforms)
    show_test_dataset = datasets.ImageFolder(opt["path_test_data_dir"], transform=show_transforms)

    print("Create data loader...")
    test_loader = DataLoader(test_dataset, batch_size=opt["batch_size"], num_workers=8)
    show_loader = DataLoader(show_test_dataset, batch_size=opt["batch_size"], num_workers=8)

    # Variables for metrics
    list_all_preds = []
    list_all_labels = []
    list_all_probs = []
    if print_class_activation_maps:
        list_all_CAMs = []

    model = model.to(device)
    gradcam_visualization = GradCam(model, device)

    print("Making predictions...")
    # Loop through the data loader
    for idx, (images, labels) in tqdm(enumerate(test_loader)):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        list_all_preds.append(preds.cpu().numpy())
        list_all_labels.append(labels.cpu().numpy())
        list_all_probs.append(F.softmax(outputs, dim=1).cpu().detach().numpy())

        if print_class_activation_maps:

            """outs = [
                gradcam_visualization(
                    images, None, postprocessing=image_net_postprocessing, target_class=preds
                )
                for c in preds
            ]"""
            cam_data = gradcam_visualization(
                images, None, postprocessing=image_net_postprocessing, target_class=preds
            )

            images, classes = vis_outs2images_classes(cam_data)

            list_all_CAMs.append(cam_data)

    for idx, (original_image, _) in tqdm(enumerate(show_loader)):

        prepared_image = np.squeeze(
            original_image.cpu().numpy()
        )  # .transpose((1, 2, 0))  # Convert to HWC format for saving
        prepared_image = prepared_image.transpose((1, 2, 0))
        prepared_image = np.clip(prepared_image * 255, 0, 255).astype(np.uint8)

        result = overlay_mask(
            to_pil_image(prepared_image),
            to_pil_image(list_all_CAMs[idx][0].squeeze(0), mode="F"),
            alpha=0.5,
        )

        # Get class names and confidence
        true_class = test_dataset.classes[int(list_all_labels[idx])]
        pred_class = test_dataset.classes[int(list_all_preds[idx])]
        confidence = list_all_probs[idx][0][int(list_all_preds[idx])]

        # Plot and save the image with true and predicted labels
        plt.figure()
        # plt.imshow(prepared_image)
        plt.imshow(result)
        plt.title(f"True: {true_class}, Pred: {pred_class} ({confidence*100:.2f}%)")
        plt.axis("off")
        output_img_path = os.path.join(opt["path_output_dir"], f"image_{idx}_pred_{pred_class}.png")
        plt.savefig(output_img_path)
        plt.close()

        # Log image to TensorBoard
        writer.add_image(f"Image_{idx}_Pred_{pred_class}", prepared_image.transpose((2, 0, 1)), 0)

    # Convert to numpy arrays
    all_preds = np.concatenate(list_all_preds)
    all_labels = np.concatenate(list_all_labels)
    all_probs = np.concatenate(list_all_probs)

    # 1. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(14, 14))
    class_plot_names = [name + "_" + str(idx) for idx, name in enumerate(test_dataset.classes)]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_plot_names,
        yticklabels=class_plot_names,
    )
    plt.show()
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    confusion_matrix_path = os.path.join(opt["path_output_dir"], "confusion_matrix.png")
    writer.add_figure("Confusion Matrix", fig, 0)
    plt.savefig(confusion_matrix_path)
    plt.close()

    # 2. Precision, Recall, F1, Average Precision (AP), and Mean Average Precision (mAP)
    precision = precision_score(all_labels, all_preds, average="macro")
    recall = recall_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")
    ap = average_precision_score(all_labels, all_probs, average="macro")

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"Average Precision (AP): {ap:.4f}")

    writer.add_scalar("Precision", precision, 0)
    writer.add_scalar("Recall", recall, 0)
    writer.add_scalar("F1 Score", f1, 0)
    writer.add_scalar("Average Precision (AP)", ap, 0)

    # 3. Precision-Recall Curve
    for i in range(opt["num_classes"]):
        precision_vals, recall_vals, _ = precision_recall_curve(all_labels == i, all_probs[:, i])
        fig, ax = plt.subplots()
        ax.plot(recall_vals, precision_vals, label=f"Class {i}")
        plt.show()
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision-Recall Curve for Class {i}")
        pr_curve_path = os.path.join(
            opt["path_output_dir"], f"precision_recall_curve_class_{i}.png"
        )
        writer.add_figure(f"Precision-Recall Curve Class {i}", fig, 0)
        plt.savefig(pr_curve_path)
        plt.close()

    # 4. ROC Curve and ROC AUC Score
    n_classes = opt["num_classes"]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(all_labels == i, all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot ROC curve
        fig, ax = plt.subplots()
        ax.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")
        ax.plot([0, 1], [0, 1], color="navy", linestyle="--")
        plt.show()
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve for Class {i}")
        roc_curve_path = os.path.join(opt["path_output_dir"], f"roc_curve_class_{i}.png")
        writer.add_figure(f"ROC Curve Class {i}", fig, 0)
        plt.savefig(roc_curve_path)
        plt.close()

    # ROC AUC Score (macro-average)
    macro_auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")
    print(f"Macro-average ROC AUC Score: {macro_auc:.4f}")
    writer.add_scalar("ROC AUC (Macro)", macro_auc, 0)

    # Save metrics to text
    with open(os.path.join(opt["path_output_dir"], "metrics.txt"), "w") as f:
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1: {f1:.4f}\n")
        f.write(f"Average Precision (AP): {ap:.4f}\n")
        f.write(f"Macro-average ROC AUC Score: {macro_auc:.4f}\n")

    # Close the TensorBoard writer
    writer.close()


if __name__ == "__main__":
    opt = get_args(
        model_type=OptionsModelType.B4,
        path_test_data_dir="/home/matthias/workspace/Coding/00_vista_medizina/00_data/2024-11-08/kaggle_bone_fracture_detection_small_2024-11-08/bf_orig_bgr_classification/test",
        path_trained_model="/home/matthias/workspace/Coding/00_vista_medizina/10_weights/efficient_net/2024-12-02_bf_kaggle/efficientnet-b4_BEST.pth",
        num_classes=7,
    )
    test(opt)