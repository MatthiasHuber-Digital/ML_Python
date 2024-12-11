from enum import Enum
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet
from torch import nn
from tqdm import tqdm
from efficient_net.options import OptionsModelType
from efficient_net.utils import load_efficient_net_model


# https://saturncloud.io/blog/how-to-solve-cuda-out-of-memory-error-in-pytorch/
scaler = torch.amp.GradScaler()
# gradient_accum_interval = 4  # A weight update is done only every 4 batches
PATH_LOAD_FROM_DISK = (
    # "/home/matthias/workspace/Coding/00_vista_medizina/vista_bone_frac/efficientnet_b4_epoch_21.pth"
    "/home/matthias/workspace/Coding/00_vista_medizina/10_weights/efficient_net/2024-12-02_bf_kaggle/efficientnet-b4_BEST.pth"
)
PATH_SAVE_TO_DISK = (
    "/home/matthias/workspace/Coding/00_vista_medizina/vista_bone_frac/efficientnet_b4_part3_"
    # "/home/matthias/workspace/Coding/00_vista_medizina/vista_bone_frac/efficientnet_b4_part2_"
)
LEARNING_RATE = 1e-7  # 1e-7
MODEL_TYPE = OptionsModelType.B4
BATCH_SIZE = 22


print("Set up device (use GPU if available)")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Paths to your train and validation directories")
# train_dir = "/home/matthias/workspace/Coding/00_vista_medizina/00_data/2024-11-08/kaggle_bone_fracture_detection_small_2024-11-08/bf_orig_bgr_classification/train"
# val_dir = "/home/matthias/workspace/Coding/00_vista_medizina/00_data/2024-11-08/kaggle_bone_fracture_detection_small_2024-11-08/bf_orig_bgr_classification/val"

print("Define image transformations for training and validation")
train_transforms = transforms.Compose(
    [
        # transforms.Resize((600, 600)),  # B7
        # transforms.Resize((380, 380)),  # B4
        transforms.Resize(MODEL_TYPE.get_resolution()),  # B4
        transforms.RandomHorizontalFlip(),  # Augmentation: random horizontal flip
        transforms.RandomVerticalFlip(),  # Augmentation: random horizontal flip
        transforms.RandomRotation(20),  # Augmentation: random rotation
        transforms.ToTensor(),  # Convert image to Tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # Normalize images
    ]
)

val_transforms = transforms.Compose(
    [
        # transforms.Resize((600, 600)),  # B7
        # transforms.Resize((380, 380)),  # B4
        transforms.Resize(MODEL_TYPE.get_resolution()),  # B4
        transforms.ToTensor(),  # Convert image to Tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # Normalize images
    ]
)

print("Load the datasets")
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

print("Create data loaders")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

print("Modify the final layer to match the number of classes in your dataset")
num_classes = len(train_dataset.classes)  # Number of classes based on the directory structure

print(f"Load the {MODEL_TYPE} model")
if PATH_LOAD_FROM_DISK is not None:
    model = load_efficient_net_model(
        path_model=PATH_LOAD_FROM_DISK,
        model_type=OptionsModelType.B4,
        num_classes=num_classes,
    )
    model.train()
else:
    model = EfficientNet.from_pretrained(MODEL_TYPE)

print("Move the model to the device (GPU or CPU)")
model = model.to(device)

print("Define the loss function and optimizer")
criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class classification
optimizer = optim.Adam(
    model.parameters(), lr=LEARNING_RATE
)  # 1e-5)  # Adam optimizer with learning rate


highest_val_acc = 0.0
print("Training loop")
epochs = 31  # Number of epochs to train
for epoch in range(1, epochs):
    print(f"Epoch [{epoch}/{epochs}]")
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    print("Training...")
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        """
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        """

        """
        Mixed precision training is a technique that uses lower-precision data types for 
        some parts of the computation to reduce memory usage and speed up training. PyTorch 
        provides support for mixed precision training through the torch.cuda.amp module.
        """
        with torch.amp.autocast(device_type=str(device)):
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
        scaler.scale(loss).backward()
        # Additionally, gradient accumulation is used for increasing the effective batch size:
        # if epoch % gradient_accum_interval == 0:
        #    scaler.step(optimizer)
        scaler.step(optimizer)
        scaler.update()
        # -------------------

        running_loss += loss.item()
        _, predicted = torch.max(output, 1)
        correct_preds += (predicted == labels).sum().item()
        total_preds += labels.size(0)

    # Calculate accuracy and loss for this epoch
    # Calculate training loss and accuracy
    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct_preds / total_preds

    # Validation phase
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    val_correct_preds = 0
    val_total_preds = 0

    print("Validation...")
    with torch.no_grad():  # No gradient calculation for validation
        for images, labels in tqdm(val_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            val_correct_preds += (predicted == labels).sum().item()
            val_total_preds += labels.size(0)

    # Calculate accuracy and loss for validation
    val_loss = val_loss / len(val_loader)
    val_acc = 100 * val_correct_preds / val_total_preds

    # Print out training and validation losses at each epoch
    print(
        f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%, "
        f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%"
    )
    if val_acc > highest_val_acc:
        highest_val_acc = val_acc
        print("==> Model saved with highest validation accuracy!")
        torch.save(model.state_dict(), PATH_SAVE_TO_DISK + "BEST.pth")
    elif epoch in [10, 20, 30, 40, 50]:
        print("==> Saving model to disk...")
        torch.save(model.state_dict(), PATH_SAVE_TO_DISK + f"{epoch}.pth")
