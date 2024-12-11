from efficientnet_pytorch import EfficientNet
import matplotlib.pyplot as plt
import os
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    roc_auc_score,
    roc_curve,
)
import torchvision.transforms.functional as F
from efficient_net.options import OptionsModelType
from efficient_net.utils import load_efficient_net_model
from efficient_net.split_dataset_kfoldcrossval import get_kfold_directory_list
from torch.utils.tensorboard import SummaryWriter

# Path and settings
PATH_LOAD_FROM_DISK = "/home/matthias/workspace/Coding/00_vista_medizina/10_weights/efficient_net/2024-12-02_bf_kaggle/efficientnet-b5_epoch30_BEST.pth"
# "/home/matthias/workspace/Coding/00_vista_medizina/10_weights/efficient_net/2024-12-02_bf_kaggle/efficientnet-b4_BEST.pth"
PATH_DATASET = "/home/matthias/workspace/Coding/00_vista_medizina/00_data/2024-11-08/kaggle_bone_fracture_detection_small_2024-11-08/bf_orig_bgr_classification_7cl_kfold/w_folds"
PATH_OUTPUT_DIR = "/home/matthias/workspace/Coding/00_vista_medizina/vista_bone_frac/efficient_net/output_7cl_5fold_3runs"

LR_INITIAL = 5e-6  # best: 1e-6
LR_PLATEAU_REDUCER = True  # const lr: False
LR_PLAT_FACTOR = 0.5  # FLOAT, pytorch: 0.1
LR_PLAT_PATIENCE = 3  # INT, pytorch:3
MODEL_TYPE = OptionsModelType.B5
BATCH_SIZE = 10
K_FOLDS = 5  # Number of folds for cross-validation
NUM_EPOCHS = 31  # Number of epochs to train
NUM_RUNS = 3  # Number of runs for random splits
LIST_SAVE_EPOCHS = [10, 20, 30]
SAVE_CURRENT_BEST_MODEL = False

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Collate function
def custom_collate_fn(batch):
    return default_collate(batch)


# Model transforms
class ResizeWithAspectRatioAndPadding:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, img):
        # Get the original dimensions of the image
        width, height = img.size
        aspect_ratio = width / height

        # Compute the new dimensions while maintaining the aspect ratio
        if aspect_ratio > 1:
            new_width = self.target_size
            new_height = int(self.target_size / aspect_ratio)
        else:
            new_height = self.target_size
            new_width = int(self.target_size * aspect_ratio)

        # Resize the image
        img = F.resize(img, (new_height, new_width))

        # Now, pad the image to match the target size
        padding = (0, 0, self.target_size - new_width, self.target_size - new_height)
        img = F.pad(img, padding, fill=0)
        return img


# Transforms for training and validation
train_transforms = transforms.Compose(
    [
        ResizeWithAspectRatioAndPadding(MODEL_TYPE.get_resolution()),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transforms = transforms.Compose(
    [
        ResizeWithAspectRatioAndPadding(MODEL_TYPE.get_resolution()),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

list_paths_kfold_dirs = get_kfold_directory_list(path_image_dir=PATH_DATASET)

# Start training process
for run in range(NUM_RUNS):
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(f"\nTRAIN RUN NUMBER: {run}/{NUM_RUNS-1}")

    # Create a directory for the current run
    run_output_dir = os.path.join(PATH_OUTPUT_DIR, f"run_{run}")
    os.makedirs(run_output_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(run_output_dir, "tensorboard"))

    # Loop over each fold
    for dict_path_dir in list(list_paths_kfold_dirs):
        idx_fold = dict_path_dir["idx_fold"]
        path_kfold_train_dir = dict_path_dir["train"]
        path_kfold_val_dir = dict_path_dir["val"]
        print(f"\n----Fold {idx_fold}----------------------------")

        print("--Creating fold output directory...")
        fold_output_dir = os.path.join(run_output_dir, f"fold_{idx_fold}")
        os.makedirs(fold_output_dir, exist_ok=True)

        print("--Creating datasets...")
        kfold_train_dataset = datasets.ImageFolder(path_kfold_train_dir, transform=train_transforms)
        kfold_val_dataset = datasets.ImageFolder(path_kfold_val_dir, transform=val_transforms)
        print("--Creating dataloaders...")
        train_fold_loader = DataLoader(
            kfold_train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            collate_fn=custom_collate_fn,
        )
        val_fold_loader = DataLoader(
            kfold_val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            collate_fn=custom_collate_fn,
        )

        print("--Loading the model...")
        if PATH_LOAD_FROM_DISK is not None:
            model = load_efficient_net_model(
                path_model=PATH_LOAD_FROM_DISK,
                model_type=MODEL_TYPE,
                num_classes=len(kfold_train_dataset.classes),
            )

        else:
            model = EfficientNet.from_pretrained(
                MODEL_TYPE, num_classes=len(kfold_train_dataset.classes)
            )

        model.to(device)

        criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss for multi-class classification
        optimizer = optim.Adam(model.parameters(), lr=LR_INITIAL)  # Adam optimizer

        scaler = torch.amp.GradScaler()
        if LR_PLATEAU_REDUCER:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=LR_PLAT_FACTOR,
                patience=LR_PLAT_PATIENCE,
                verbose=True,
            )

        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []

        highest_val_acc = 0.0
        print("--Starting the training process...")
        for epoch in range(NUM_EPOCHS):
            print(
                f"\n+++++ Epoch [{epoch}/{NUM_EPOCHS}] ++++++++++++++++++++++++++++++++++++++++++++++"
            )
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"--learning rate: {current_lr:.1e}")

            # Training Phase
            model.train()
            running_loss = 0.0
            correct_preds = 0
            total_preds = 0
            all_preds = []
            all_labels = []

            print("--training loop")
            for images, labels in tqdm(train_fold_loader):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.amp.autocast(device_type=str(device)):  # Mixed precision training
                    output = model(images)
                    loss = criterion(output, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                _, predicted = torch.max(output, 1)
                correct_preds += (predicted == labels).sum().item()
                total_preds += labels.size(0)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            # Calculate training metrics
            train_loss = running_loss / len(train_fold_loader)
            train_acc = 100 * correct_preds / total_preds
            f1 = f1_score(all_labels, all_preds, average="weighted")
            precision = precision_score(all_labels, all_preds, average="weighted")
            recall = recall_score(all_labels, all_preds, average="weighted")
            average_precision = average_precision_score(all_labels, all_preds, average="weighted")

            # Log to TensorBoard
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            writer.add_scalar("F1/train", f1, epoch)
            writer.add_scalar("Precision/train", precision, epoch)
            writer.add_scalar("Recall/train", recall, epoch)
            writer.add_scalar("Average Precision/train", average_precision, epoch)

            # Validation
            model.eval()
            val_loss, val_correct_preds, val_total_preds = 0.0, 0, 0
            print("--validation loop")
            val_all_preds, val_all_labels = [], []
            with torch.no_grad():
                for images, labels in tqdm(val_fold_loader):
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_correct_preds += (predicted == labels).sum().item()
                    val_total_preds += labels.size(0)

                    val_all_preds.extend(predicted.cpu().numpy())
                    val_all_labels.extend(labels.cpu().numpy())

            val_loss = val_loss / len(val_fold_loader)
            val_acc = 100 * val_correct_preds / val_total_preds
            val_f1 = f1_score(val_all_labels, val_all_preds, average="weighted")
            val_precision = precision_score(val_all_labels, val_all_preds, average="weighted")
            val_recall = recall_score(val_all_labels, val_all_preds, average="weighted")
            val_avg_precision = average_precision_score(
                val_all_labels, val_all_preds, average="weighted"
            )

            # Log validation metrics to TensorBoard
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)
            writer.add_scalar("F1/val", val_f1, epoch)
            writer.add_scalar("Precision/val", val_precision, epoch)
            writer.add_scalar("Recall/val", val_recall, epoch)
            writer.add_scalar("Average Precision/val", val_avg_precision, epoch)
            if LR_PLATEAU_REDUCER:
                scheduler.step(val_loss)
                print(
                    f"--LR scheduler: ReduceLROnPlateau reducing learning rate to {scheduler.get_last_lr()[0]:.1e}"
                )

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)

            # Print out training and validation losses and accuracies
            print(
                f"---- train loss: {train_loss:.4f}, train accuracy: {train_acc:.2f}%, "
                f"---- validation loss: {val_loss:.4f}, validation accuracy: {val_acc:.2f}%"
            )

            # Save model with highest validation accuracy
            if (
                SAVE_CURRENT_BEST_MODEL
                and epoch > int(NUM_EPOCHS * 0.5)
                and val_acc > highest_val_acc
            ):
                highest_val_acc = val_acc
                print(f"Saving model with highest validation accuracy {val_acc:.2f}%")
                torch.save(
                    model.state_dict(),
                    os.path.join(fold_output_dir, f"model_epoch_{epoch}_BEST.pth"),
                )

            # Save the model at specified epochs
            if (
                LIST_SAVE_EPOCHS is not None
                and len(LIST_SAVE_EPOCHS) > 0
                and epoch in LIST_SAVE_EPOCHS
            ):
                print("==> Saving model to disk...")
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        fold_output_dir, str(MODEL_TYPE) + "_epoch" + str(epoch) + "_BEST.pth"
                    ),
                )
            # Free up memory after each epoch
            del images, labels, output, loss
            torch.cuda.empty_cache()

        print(
            "--finished computing fold number " + str(idx_fold) + ", run number " + str(run) + "."
        )
        # Plot and save the loss curves
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(NUM_EPOCHS), train_losses, label="Training Loss")
        plt.plot(range(NUM_EPOCHS), val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(range(NUM_EPOCHS), train_accuracies, label="Training Accuracy")
        plt.plot(range(NUM_EPOCHS), val_accuracies, label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(fold_output_dir, "loss_accuracy_curve.png"))
        plt.close()
        print("--finished saving loss plots.")

        writer.flush()
        print(f"Finished fold {idx_fold} of run {run + 1}/{NUM_RUNS}.")
        writer.close()

    print(f"Finished run {run + 1}/{NUM_RUNS}.")
