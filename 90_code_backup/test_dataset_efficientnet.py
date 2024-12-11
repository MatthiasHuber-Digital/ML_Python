import matplotlib.pyplot as plt
import numpy as np
from omegaconf.omegaconf import OmegaConf
import os
import shutil
from sklearn.metrics import *
import seaborn as sns
import shutil
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from options import OptionsModelType
from utils import load_efficient_net_model


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
    }

    args = OmegaConf.create(dict_args)

    return args


"""
def test(opt):
    model = load_efficient_net_model(
        path_model=opt.pretrained_model,
        model_type=opt.model_type,
        num_classes=opt.num_classes,
    )
    model.cuda().eval()
    test_transforms = transforms.Compose(
        [
            transforms.Resize(opt.model_type.get_resolution()),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    print("Create the output directory: %s" % opt.path_output_dir)
    if os.path.isdir(opt.output):
        shutil.rmtree(opt.output)
    os.makedirs(opt.output)

    print("Load the datasets")
    test_dataset = datasets.ImageFolder(opt.data_path, transform=test_transforms)

    print("Create data loader...")
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, num_workers=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Making predictions...")
    for images, labels in tqdm(test_loader):
        images, labels = images.to(device), labels.to(device)
"""


def test(opt):
    # Load the model
    model = load_efficient_net_model(
        path_model=opt.path_trained_model,
        model_type=opt.model_type,
        num_classes=opt.num_classes,
    )
    model.cuda().eval()

    # Data transformations
    test_transforms = transforms.Compose(
        [
            transforms.Resize(opt.model_type.get_resolution()),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create the output directory
    print("Create the output directory (or overwrite the existing one): %s" % opt.path_output_dir)
    if os.path.isdir(opt.path_output_dir):
        shutil.rmtree(opt.path_output_dir)
    os.makedirs(opt.path_output_dir)

    # Initialize TensorBoard writer
    writer = SummaryWriter(opt.path_output_dir)

    # Load dataset and dataloader
    print("Load the datasets")
    test_dataset = datasets.ImageFolder(opt.path_test_data_dir, transform=test_transforms)
    show_test_dataset = datasets.ImageFolder(opt.path_test_data_dir, transform=None)

    print("Create data loader...")
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, num_workers=8)
    show_loader = DataLoader(show_test_dataset, batch_size=opt.batch_size, num_workers=8)

    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Variables for metrics
    list_all_preds = []
    list_all_labels = []
    list_all_probs = []

    print("Making predictions...")

    # Loop through the data loader
    for images, labels in tqdm(test_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        # Predicted labels
        _, preds = torch.max(outputs, 1)
        list_all_preds.append(preds.cpu().numpy())
        list_all_labels.append(labels.cpu().numpy())
        list_all_probs.append(F.softmax(outputs, dim=1).cpu().detach().numpy())

    # Convert to numpy arrays
    all_preds = np.concatenate(list_all_preds)
    all_labels = np.concatenate(list_all_labels)
    all_probs = np.concatenate(list_all_probs)

    # 1. Log Images with Predictions
    img_grid = torchvision.utils.make_grid(
        images[: len(test_loader)]
    )  # Convert batch of images to grid

    def log_images_with_predictions(images, labels, preds, writer, n=opt.batch_size):
        writer.add_image("Images", img_grid, 0)
        # Log predicted and actual labels for the first few images
        for i in range(n):
            writer.add_text(f"Image_{i}_True_Label", f"{test_dataset.classes[labels[i]]}", 0)
            writer.add_text(f"Image_{i}_Pred_Label", f"{test_dataset.classes[preds[i]]}", 0)

    # Get a batch of images to log
    for images, labels in tqdm(test_loader):
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1).cpu().numpy()
        log_images_with_predictions(list(images), list(labels.cpu().numpy()), list(preds), writer)

    # 2. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(14, 14))
    # class_plot_indices = [idx for idx in range(len(test_dataset.classes))]
    class_plot_names = [name + "_" + str(idx) for idx, name in enumerate(test_dataset.classes)]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_plot_names,
        yticklabels=class_plot_names,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    writer.add_figure("Confusion Matrix", fig, 0)
    plt.close()

    # 3. Precision, Recall, F1, Average Precision (AP), and Mean Average Precision (mAP)
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

    # 4. ROC Curve and ROC AUC Score
    n_classes = opt.num_classes
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(all_labels == i, all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve for each class
    fig, ax = plt.subplots()
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")

    ax.plot([0, 1], [0, 1], color="navy", linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.legend(loc="lower right")
    writer.add_figure("ROC Curve", fig, 0)
    plt.close()

    # ROC AUC Score (macro-average)
    macro_auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")
    print(f"Macro-average ROC AUC Score: {macro_auc:.4f}")
    writer.add_scalar("ROC AUC (Macro)", macro_auc, 0)

    # Save metrics to text
    with open(os.path.join(opt.path_output_dir, "metrics.txt"), "w") as f:
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
