from efficientnet_pytorch import EfficientNet
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from efficient_net.options import OptionsModelType
from efficient_net.utils import load_efficient_net_model
from utils.plotting import plot_class_distribution


def get_args(
    model_type: OptionsModelType,
    path_test_data_dir: str,
    path_trained_model: str,
    list_class_plot_names: list[str],
):
    dict_args = {
        "model_type": model_type,
        "path_test_data_dir": path_test_data_dir,
        "path_trained_model": path_trained_model,
        "num_classes": len(list_class_plot_names),
        "path_output_dir": "predictions",
        "batch_size": 1,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "print_class_activation_maps": True,
        "list_class_plot_names": list_class_plot_names,
    }

    # args = OmegaConf.create(dict_args)

    return dict_args


def compute_and_plot_confusion_matrix(
    np_gt_classes: np.ndarray,
    np_predicted_classes: np.ndarray,
    list_class_names: list[str],
    path_output_dir: str,
    tensorboard_writer: SummaryWriter = None,  # This is the Tensorboard writer object
):
    """This function computes the confusion matrix, plots and saves it to the output directory.

    Args:
        np_gt_classes (np.ndarray): A list of true labels for the data.
        np_predicted_classes (np.ndarray): A list of predicted labels for the data.
        list_class_names (list[str]): A list of class names corresponding to the labels.
        path_output_dir (str): A string representing the path to the output directory where the confusion matrix plot will be saved.
        tensorboard_writer (SummaryWriter): A SummaryWriter object for Tensorboard.
    """
    cm = confusion_matrix(np_gt_classes, np_predicted_classes)
    fig, ax = plt.subplots(figsize=(14, 14))

    # list_class_indices = [name[-1] for name in list_class_names]

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=list_class_names,
        yticklabels=list_class_names,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.savefig(os.path.join(path_output_dir, "confusion_matrix.png"))
    plt.show()

    if tensorboard_writer is not None:
        tensorboard_writer.add_figure("Confusion Matrix", fig, 0)

    plt.close()


def compute_classification_metrics(
    np_gt_classes: np.ndarray,
    np_predicted_classes: np.ndarray,
    np_pred_probs: np.ndarray,
    num_classes: int,
    tensorboard_writer: SummaryWriter = None,
) -> dict:
    """
    Compute and log various classification metrics.

    This function calculates precision, recall, F1 score, and average precision
    for the given true labels and predictions. It prints these metrics and
    logs them to TensorBoard if a writer is provided.

    Args:
        np_gt_classes (np.ndarray): A list of true labels for the data.
        np_predicted_classes (np.ndarray): A list of predicted labels for the data.
        np_pred_probs (np.ndarray): A list of prediction probabilities for all classes of the data.
        num_classes (int): Number of classes.
        tensorboard_writer (SummaryWriter, optional): A TensorBoard SummaryWriter
            object for logging the metrics. Defaults to None.

    Returns:
        dict: A dictionary containing the computed metrics:
            - 'precision': The macro-averaged precision score.
            - 'recall': The macro-averaged recall score.
            - 'f1': The macro-averaged F1 score.
            - 'ap': The macro-averaged average precision score.
    """
    precision = precision_score(np_gt_classes, np_predicted_classes, average="macro")
    recall = recall_score(np_gt_classes, np_predicted_classes, average="macro")
    f1 = f1_score(np_gt_classes, np_predicted_classes, average="macro")
    if num_classes > 2:
        ap = average_precision_score(np_gt_classes, np_pred_probs, average="macro")
    else:
        ap = precision

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"Average Precision (AP): {ap:.4f}")

    tensorboard_writer.add_scalar("Precision", precision, 0)
    tensorboard_writer.add_scalar("Recall", recall, 0)
    tensorboard_writer.add_scalar("F1 Score", f1, 0)
    tensorboard_writer.add_scalar("Average Precision (AP)", ap, 0)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ap": ap,
    }


def compute_and_plot_precision_recall_curves(
    np_gt_classes: np.ndarray,
    np_pred_probs: np.ndarray,
    list_class_names: list[str],
    path_output_dir: str,
    tensorboard_writer: SummaryWriter = None,
):
    """
    Compute and plot precision-recall curves for each class.

    This function calculates precision-recall curves for each class, plots them,
    saves the plots to the specified output directory, and optionally logs them to TensorBoard.

    Args:
        np_gt_classes (np.ndarray): True labels for each sample.
        np_pred_probs (np.ndarray): A list of prediction probabilities for all classes of the data.
        list_class_names (list[str]): Names of the classes.
        path_output_dir (str): Directory path to save the output plots.
        tensorboard_writer (SummaryWriter, optional): TensorBoard SummaryWriter object for logging. Defaults to None.

    Returns:
        None
    """
    for i_gt_class, gt_class_name in enumerate(list_class_names):

        """# Mark sample indices carrying the GT-class of gt_class_name:
        list_gt_class_sample_indices = list(
            np.where(np_gt_classes == i_gt_class)[0]
        )  # indices of predictions where the gt class was the current one
        list_preds_to_gt_samples = list(np_pred_probs)
        # Get the prediction probabilities for ALL classes for the pictures that have GT class "gt_class_name":
        list_preds_to_gt_samples = [
            data_row
            for idx_row, data_row in enumerate(list_preds_to_gt_samples)
            if idx_row in list_gt_class_sample_indices
        ]
        # Extract ONLY the probabilities corresponding to the real GT class:
        np_pred_probs_of_gt_class = np.array(list_preds_to_gt_samples)[:, i_gt_class]

        # An np array with positive class "1" (i.e. "true") of length of the number of GT class samples is passed as y_true:
        precision_vals, recall_vals, _ = precision_recall_curve(
            y_true=np.ones(len(list_gt_class_sample_indices)),
            y_score=np_pred_probs_of_gt_class,
        )"""
        precision_vals, recall_vals, _ = precision_recall_curve(
            np_gt_classes == i_gt_class, np_pred_probs[:, i_gt_class]
        )
        fig, ax = plt.subplots()
        ax.plot(recall_vals, precision_vals, label=f"Class {gt_class_name}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision-Recall Curve for Class {gt_class_name}")
        plt.savefig(
            os.path.join(path_output_dir, f"precision_recall_curve_class_{gt_class_name}.png")
        )
        plt.show()

        if tensorboard_writer is not None:
            tensorboard_writer.add_figure(f"Precision-Recall Curve Class {gt_class_name}", fig, 0)

        plt.close()


def compute_and_plot_roc_auc_curves(
    np_gt_classes: np.ndarray,
    np_pred_probs: np.ndarray,
    list_class_names: list[str],
    path_output_dir: str,
    tensorboard_writer: SummaryWriter = None,
):
    """
    Compute and plot ROC AUC curves for each class and calculate the macro-average ROC AUC score.

    This function calculates the Receiver Operating Characteristic (ROC) curve and Area Under the Curve (AUC)
    for each class, plots these curves, saves them to the specified output directory, and optionally logs
    them to TensorBoard. It also computes and logs the macro-average ROC AUC score.

    Args:
        list_gt_classes (np.ndarray): True labels for each sample.
        list_probs (np.ndarray): Predicted probabilities for each class and sample.
        list_class_names (list[str]): Names of the classes.
        path_output_dir (str): Directory path to save the output plots.
        tensorboard_writer (SummaryWriter, optional): TensorBoard SummaryWriter object for logging. Defaults to None.

    Returns:
        macro_auc (float): Macro-average ROC AUC score.
    """
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i, gt_class_name in enumerate(list_class_names):
        fpr[i], tpr[i], _ = roc_curve(np_gt_classes == i, np_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot ROC curve
        fig, ax = plt.subplots()
        ax.plot(fpr[i], tpr[i], label=f"Class {gt_class_name} (AUC = {roc_auc[i]:.2f})")
        ax.plot([0, 1], [0, 1], color="navy", linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve for Class {gt_class_name}")
        plt.savefig(os.path.join(path_output_dir, f"roc_curve_class_{gt_class_name}.png"))
        plt.show()

        if tensorboard_writer is not None:
            tensorboard_writer.add_figure(f"ROC Curve Class {gt_class_name}", fig, 0)

        plt.close()

    if len(list_class_plot_names) > 2:
        macro_auc = roc_auc_score(np_gt_classes, np_pred_probs, multi_class="ovr")
        print(f"Macro-average ROC AUC Score: {macro_auc:.4f}")
        if tensorboard_writer is not None:
            tensorboard_writer.add_scalar("ROC AUC (Macro)", macro_auc, 0)
        return macro_auc
    else:
        roc_auc = auc(fpr[0], tpr[0])
        print(f"ROC AUC Score: {roc_auc:.4f}")
        if tensorboard_writer is not None:
            tensorboard_writer.add_scalar("ROC AUC", roc_auc, 0)
        return roc_auc


def test(opt):
    # Load the model
    model = load_efficient_net_model(
        path_model=opt["path_trained_model"],
        model_type=opt["model_type"],
        num_classes=opt["num_classes"],
    )
    model.cuda().eval()

    # Data transformations
    test_transforms = transforms.Compose(
        [
            transforms.Resize(opt["model_type"].get_resolution()),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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

    with PdfPages(os.path.join(opt["path_output_dir"], f"test_eval_report.pdf")) as pdf_pages:
        plot_class_distribution(
            list_class_occurences=test_dataset.__dict__["targets"],
            save_object=pdf_pages,
            title="Class Distribution",
        )

    print("Create data loader...")
    test_loader = DataLoader(test_dataset, batch_size=opt["batch_size"], num_workers=8)

    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Variables for metrics
    list_all_predicted_classes = []
    list_all_gt_classes = []
    list_all_pred_probs = []

    print("Making predictions...")
    # Loop through the data loader
    for _, (images, labels) in tqdm(enumerate(test_loader)):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted_class = torch.max(outputs, 1)

        list_all_predicted_classes.append(predicted_class.cpu().numpy())
        list_all_gt_classes.append(labels.cpu().numpy())
        list_all_pred_probs.append(F.softmax(outputs, dim=1).cpu().detach().numpy())

    # Convert to numpy arrays
    np_predicted_classes = np.concatenate(list_all_predicted_classes)
    np_gt_classes = np.concatenate(list_all_gt_classes)
    np_pred_probs = np.concatenate(list_all_pred_probs)

    compute_and_plot_confusion_matrix(
        np_gt_classes=np_gt_classes,
        np_predicted_classes=np_predicted_classes,
        list_class_names=list_class_plot_names,
        path_output_dir=opt["path_output_dir"],
        tensorboard_writer=writer,
    )
    dict_metrics = compute_classification_metrics(
        np_gt_classes=np_gt_classes,
        np_predicted_classes=np_predicted_classes,
        np_pred_probs=np_pred_probs,
        num_classes=opt["num_classes"],
        tensorboard_writer=writer,
    )
    compute_and_plot_precision_recall_curves(
        np_gt_classes=np_gt_classes,
        np_pred_probs=np_pred_probs,
        list_class_names=list_class_plot_names,
        path_output_dir=opt["path_output_dir"],
        tensorboard_writer=writer,
    )
    macro_auc = compute_and_plot_roc_auc_curves(
        np_gt_classes=np_gt_classes,
        np_pred_probs=np_pred_probs,
        list_class_names=list_class_plot_names,
        path_output_dir=opt["path_output_dir"],
        tensorboard_writer=writer,
    )

    with open(os.path.join(opt["path_output_dir"], "metrics.txt"), "w") as f:
        precision = dict_metrics["precision"]
        recall = dict_metrics["recall"]
        f1 = dict_metrics["f1"]
        ap = dict_metrics["ap"]
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1: {f1:.4f}\n")
        f.write(f"Average Precision (AP): {ap:.4f}\n")
        f.write(f"Macro-average ROC AUC Score: {macro_auc:.4f}\n")

    # Close the TensorBoard writer
    writer.close()


if __name__ == "__main__":
    """
    list_class_plot_names = [
        "elbow_0",
        "fingers_1",
        "forearm_2",
        "humerus_3",
        "none_4",  # no_fractures
        "shoulder_5",
        "wrist_6",
    ]
    """
    list_class_plot_names = [
        "fracture",
        "no_fracture",
    ]

    opt = get_args(
        model_type=OptionsModelType.B4,
        # path_test_data_dir="/home/matthias/workspace/Coding/00_vista_medizina/00_data/2024-11-22_vista_test/ba",
        # path_test_data_dir="/home/matthias/workspace/Coding/00_vista_medizina/00_data/2024-11-22_vista_test/test_set_selected/selected",
        # path_test_data_dir="/home/matthias/workspace/Coding/00_vista_medizina/00_data/2024-11-08/kaggle_bone_fracture_detection_small_2024-11-08/bf_orig_bgr_classification_7cl_kfold/w_folds/test",
        path_test_data_dir="/home/matthias/workspace/Coding/00_vista_medizina/00_data/2024-11-08/kaggle_bone_fracture_detection_small_2024-11-08/bf_orig_bgr_classification_2cl/kfold/test",
        path_trained_model="/home/matthias/workspace/Coding/00_vista_medizina/10_weights/efficient_net/2024-12-09_bf_kaggle_2cl_85pctval/efficientnet-b4_fold3_epoch30_SCHED.pth",
        list_class_plot_names=list_class_plot_names,
    )
    test(opt)
