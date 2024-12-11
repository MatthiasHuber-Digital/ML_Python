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
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from options import OptionsModelType
from utils import load_efficient_net_model


def hook_fn(module, input, output):
    """This function creates a hook inside the module, making the feature maps of the layer globally available.

    It is to be used e.g. for class activation mapping (CAM).

    Args:
        module (_type_): _description_
        input (_type_): _description_
        output (_type_): _description_
    """
    global feature_maps
    feature_maps = output


def compute_class_activation_mapping(
    model: EfficientNet, image_data: np.ndarray, idx_pred_class: int
) -> np.ndarray:
    """This function computes the class activation mappings.

    Args:
        model (EfficientNet): The EfficientNet model instance.
        idx_pred_class (int): The index of the predicted class.
        image_data (np.ndarray): The input image data.

    Returns:
        np.ndarray: Class activation mappings for the target class.
    """
    # Get the weights of the FC layer
    fc_weights = model.classifier[1].weight[idx_pred_class].detach().cpu().numpy()

    # Global average pooling of the feature maps
    pooled_feature_maps = torch.mean(feature_maps, dim=(2, 3))[0].detach()

    # Weighted sum of feature maps
    cam = torch.sum(pooled_feature_maps * torch.tensor(fc_weights), dim=0)

    # Apply ReLU to the class activation map
    cam = torch.relu(cam)

    # Normalize the CAM for visualization
    cam = cam - cam.min()
    cam = cam / cam.max()

    # Resize CAM to the image size
    cam = torch.nn.functional.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=(image_data.height, image_data.width),
        mode="bilinear",
        align_corners=False,
    )

    return cam.squeeze().cpu().numpy()


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
    model.cuda().eval()

    print_class_activation_maps = opt["print_class_activation_maps"]
    if print_class_activation_maps:
        # cam_extractor = SmoothGradCAMpp(model, "_fc")
        # Hook to get the output of the last convolutional layer
        # Register the hook to the last convolutional layer
        last_conv_layer = model._conv_head  # features[-1] # Last conv layer in EfficientNet-B0
        # hook = last_conv_layer.register_forward_hook(hook_fn)

        # Backpropagate to get gradients of the target class
        # model.zero_grad()
        # output[0, predicted_class].backward()
        def generate_cam(input_batch, model, last_conv_layer):

            hook = last_conv_layer.register_forward_hook(hook_fn)

            with torch.no_grad():
                output = model(input_batch)

            _, predicted_class = output.max(1)
            model.zero_grad()
            output[:, predicted_class].backward()  # 0, predicted_class].backward()
            fc_weights = model.classifier[1].weight[predicted_class].detach().cpu().numpy()
            pooled_feature_maps = torch.mean(feature_maps, dim=(2, 3))[0].detach()

            # Weighted sum of feature maps to generate CAM
            cam = torch.sum(pooled_feature_maps * torch.tensor(fc_weights), dim=0)
            # Apply ReLU to the CAM (to focus on positive contributions)
            cam = torch.relu(cam)
            # Normalize the CAM for visualization
            cam = cam - cam.min()
            cam = cam / cam.max()

            # Resize the CAM to match the image size
            cam = torch.nn.functional.interpolate(
                cam.unsqueeze(0).unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
            )
            cam = cam.squeeze().cpu().numpy()

            return cam, predicted_class.item()

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

    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Variables for metrics
    list_all_preds = []
    list_all_labels = []
    list_all_probs = []
    if print_class_activation_maps:
        list_all_CAMs = []

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

            cam, _ = generate_cam(images, model, last_conv_layer)
            list_all_CAMs.append(cam)

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
        path_test_data_dir="/home/matthias/workspace/Coding/00_vista_medizina/00_data/2024-11-08/kaggle_bone_fracture_detection_small_2024-11-08/bf_orig_bgr_classification_7cl/test",
        path_trained_model="/home/matthias/workspace/Coding/00_vista_medizina/10_weights/efficient_net/2024-12-02_bf_kaggle/efficientnet-b4_BEST.pth",
        num_classes=7,
    )
    test(opt)
