import os
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torch import nn
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    auc,
)
import matplotlib.pyplot as plt
from efficientnet_pytorch import EfficientNet
from efficient_net.options import OptionsModelType
from efficient_net.utils import load_efficient_net_model
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import DatasetFolder
from PIL import Image
import numpy as np

# Path and settings
PATH_LOAD_FROM_DISK = "/home/matthias/workspace/Coding/00_vista_medizina/10_weights/efficient_net/2024-12-02_bf_kaggle/efficientnet-b4_BEST.pth"
PATH_SAVE_TO_DISK = "/home/matthias/workspace/Coding/00_vista_medizina/vista_bone_frac/efficientnet_b4_kfcrossval_varruns_"
LEARNING_RATE = 1e-7  # 1e-7
MODEL_TYPE = OptionsModelType.B4
BATCH_SIZE = 22

TRAIN_TEST_SPLIT_RATIO = 0.95  # this is the amount of TRAIN/VAL data, the rest is test data
K_FOLDS = 5  # Number of folds for cross-validation
EPOCHS = 31  # Number of epochs to train
NUM_RUNS = 3  # Number of runs for random splits

# Device setup (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset directory
dataset_dir = "/home/matthias/workspace/Coding/00_vista_medizina/00_data/2024-11-08/kaggle_bone_fracture_detection_small_2024-11-08/bf_orig_bgr_classification_7cl_kfold/all"

# Define image transformations for training and validation
train_transforms = transforms.Compose(
    [
        transforms.Resize(MODEL_TYPE.get_resolution()),  # Resize to model's input resolution
        transforms.RandomHorizontalFlip(),  # Augmentation: random horizontal flip
        transforms.RandomVerticalFlip(),  # Augmentation: random vertical flip
        transforms.RandomRotation(20),  # Augmentation: random rotation
        transforms.ToTensor(),  # Convert image to Tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # Normalize images
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize(MODEL_TYPE.get_resolution()),  # Resize to model's input resolution
        transforms.ToTensor(),  # Convert image to Tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # Normalize images
    ]
)


# Custom loader function for DatasetFolder (you can keep this, or use default)
def loader(path):
    return Image.open(path).convert("RGB")

# Load the dataset using DatasetFolder without applying any transforms initially
dataset = DatasetFolder(
    root=dataset_dir,
    loader=loader,
    extensions=(".jpg", ".png"),  # Assuming your dataset contains .jpg or .png images
    transform=None,  # Don't apply any transforms initially
)

# Get class-to-label mapping
class_to_idx = dataset.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}

# Run the training process 3 times with random splits each time
for run in range(NUM_RUNS):
    print(f"\nRunning training: {run + 1}/{NUM_RUNS}")

    # Initialize TensorBoard writer
    writer = SummaryWriter(f"logs/run_{run + 1}")

    # Shuffle the dataset indices
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    random.shuffle(indices)

    # Define the split ratio for training and testing
    train_size = int(TRAIN_TEST_SPLIT_RATIO * dataset_size)
    test_size = dataset_size - train_size

    # Split dataset into training and testing
    train_indices, test_indices = indices[:train_size], indices[train_size:]

    # Create the training and test subsets
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)

    # Create DataLoader for training and testing with appropriate transforms
    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        collate_fn=None,
    )

    # Apply val_transforms for the test dataset
    test_loader = DataLoader(
        DatasetFolder(
            root=dataset_dir,
            loader=loader,
            extensions=(".jpg", ".png"),
            transform=val_transforms,  # Use val_transforms for testing
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
    )

    # K-Fold Cross Validation setup
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    # Create the k-fold splits and loop over them
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_subset)):
        print(f"\nFold {fold + 1}/{K_FOLDS}")

        # Create Subsets for Training and Validation
        train_fold_subset = Subset(train_subset, train_idx)
        val_fold_subset = Subset(train_subset, val_idx)

        # Create DataLoaders for Training and Validation using appropriate transforms
        train_fold_loader = DataLoader(
            train_fold_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8
        )

        # Apply val_transforms for the validation set
        val_fold_loader = DataLoader(
            DatasetFolder(
                root=dataset_dir,
                loader=loader,
                extensions=(".jpg", ".png"),
                transform=val_transforms,  # Use val_transforms for validation
            ),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=8,
        )

        # Load model
        if PATH_LOAD_FROM_DISK is not None:
            model = load_efficient_net_model(
                path_model=PATH_LOAD_FROM_DISK,
                model_type=OptionsModelType.B4,
                num_classes=len(dataset.classes),
            )
            model.train()
        else:
            model = EfficientNet.from_pretrained(MODEL_TYPE, num_classes=len(dataset.classes))

        model.to(device)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss for multi-class classification
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Adam optimizer

        # Initialize lists to store metrics for the current fold
        all_preds = []
        all_labels = []

        # Training and Validation Loop for the current fold
        highest_val_acc = 0.0
        for epoch in range(1, EPOCHS + 1):
            print(f"\nEpoch [{epoch}/{EPOCHS}]")

            # Training Phase
            model.train()
            running_loss = 0.0
            correct_preds = 0
            total_preds = 0

            print("Training...")
            for images, labels in tqdm(train_fold_loader):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.amp.autocast(device_type=str(device)):  # Mixed precision training
                    output = model(images)
                    loss = criterion(output, labels)

                scaler = torch.cuda.amp.GradScaler()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                _, predicted = torch.max(output, 1)
                correct_preds += (predicted == labels).sum().item()
                total_preds += labels.size(0)

            # Calculate training loss and accuracy
            train_loss = running_loss / len(train_fold_loader)
            train_acc = 100 * correct_preds / total_preds

            # Validation Phase
            model.eval()
            val_loss = 0.0
            val_correct_preds = 0
            val_total_preds = 0

            print("Validation...")
            with torch.no_grad():  # No gradient calculation during validation
                for images, labels in tqdm(val_fold_loader):
                    images, labels = images.to(device), labels.to(device)

                    output = model(images)
                    loss = criterion(output, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(output, 1)
                    val_correct_preds += (predicted == labels).sum().item()
                    val_total_preds += labels.size(0)

                    # Collect all predictions and labels for post-processing
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            # Calculate validation loss and accuracy
            val_loss = val_loss / len(val_fold_loader)
            val_acc = 100 * val_correct_preds / val_total_preds

            # Print out training and validation losses and accuracies
            print(
                f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%, "
                f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%"
            )

            # Log to TensorBoard
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)

            # Save model with highest validation accuracy
            if val_acc > highest_val_acc:
                highest_val_acc = val_acc
                torch.save(model.state_dict(), f"{PATH_SAVE_TO_DISK}fold_{fold + 1}_best_model.pth")

        # Post-process and save metrics for each fold
        # Compute additional metrics (F1, Precision, Recall, AUC, etc.)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Compute F1, Precision, Recall, ROC-AUC, Confusion Matrix, etc.
        f1 = f1_score(all_labels, all_preds, average="weighted")
        precision = precision_score(all_labels, all_preds, average="weighted")
        recall = recall_score(all_labels, all_preds, average="weighted")
        roc_auc = roc_auc_score(all_labels, all_preds, multi_class="ovr")
        conf_matrix = confusion_matrix(all_labels, all_preds)

        # Log metrics to TensorBoard
        writer.add_scalar("Metrics/F1", f1, epoch)
        writer.add_scalar("Metrics/Precision", precision, epoch)
        writer.add_scalar("Metrics/Recall", recall, epoch)
        writer.add_scalar("Metrics/ROC-AUC", roc_auc, epoch)

        # Save confusion matrix plot
        plt.figure(figsize=(8, 6))
        plt.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        plt.xticks(range(len(class_to_idx)), class_to_idx.keys(), rotation=45)
        plt.yticks(range(len(class_to_idx)), class_to_idx.keys())
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.tight_layout()
        plt.savefig(f"confusion_matrix_fold_{fold + 1}.png")

    writer.close()  # Close the TensorBoard writer after each run
