import torch
import numpy as np


def calculate_iou(
    box1: list | tuple,
    box2: list | tuple,
) -> float:
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1: Tensor of shape (4,) representing [x1, y1, x2, y2] for the first box
        box2: Tensor of shape (4,) representing [x1, y1, x2, y2] for the second box

    Returns:
        IoU: Intersection over union between the two boxes.
    """
    # Calculate intersection
    x1 = torch.max(box1[0], box2[0])
    y1 = torch.max(box1[1], box2[1])
    x2 = torch.min(box1[2], box2[2])
    y2 = torch.min(box1[3], box2[3])

    # Calculate intersection area
    inter_area = torch.max(x2 - x1, torch.tensor(0.0)) * torch.max(
        y2 - y1, torch.tensor(0.0)
    )

    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area

    return iou


def calculate_ap(
    detections: list[tuple[torch.Tensor, float]],
    ground_truths: list[torch.Tensor],
    iou_threshold: float = 0.5,
) -> float:
    """
    Calculate Average Precision at a given IoU threshold.

    Args:
        detections: List of tuples [(bbox, score), ...] for each image, where:
            - bbox is [x1, y1, x2, y2] (Tensor of size 4)
            - score is the confidence score for the detection (Tensor)
        ground_truths: List of tensors for each image, containing ground truth bounding boxes (Tensor of size [N, 4])
        iou_threshold: IoU threshold to consider a detection as a True Positive (default=0.5)

    Returns:
        average_precision: Computed average precision at the given IoU threshold
    """
    # Sort detections by score in descending order
    detections = sorted(detections, key=lambda x: x[1], reverse=True)

    true_positives = []
    false_positives = []
    scores = []

    for detection, score in detections:
        # Compute IoU with all ground truth boxes
        iou_values = [calculate_iou(detection[0], gt) for gt in ground_truths]
        max_iou = torch.max(torch.tensor(iou_values))  # Max IoU with any ground truth

        if max_iou >= iou_threshold:
            true_positives.append(1)  # True positive if IoU > threshold
            false_positives.append(0)
            # Remove this ground truth from the pool (as it's been matched)
            ground_truths = [
                gt for gt, iou in zip(ground_truths, iou_values) if iou < max_iou
            ]
        else:
            true_positives.append(0)
            false_positives.append(1)

        scores.append(score)

    # Compute precision and recall
    true_positives = np.cumsum(true_positives)
    false_positives = np.cumsum(false_positives)
    recall = true_positives / len(ground_truths)
    precision = true_positives / (true_positives + false_positives)

    # Compute average precision (AP) using the precision-recall curve
    # AP is the area under the precision-recall curve, here we use simple trapezoidal rule
    recall = np.concatenate(([0], recall, [1]))
    precision = np.concatenate(([0], precision, [0]))

    # Interpolating precision: ensure precision is always non-decreasing
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    # Calculate AP as the area under the curve using the trapezoidal rule
    ap = np.sum((recall[1:] - recall[:-1]) * precision[1:])

    return ap


# Example usage
detections = [
    (torch.tensor([50, 50, 150, 150]), 0.9),  # (bbox, score)
    (torch.tensor([30, 30, 130, 130]), 0.75),
    (torch.tensor([60, 60, 160, 160]), 0.85),
]
ground_truths = [torch.tensor([50, 50, 150, 150]), torch.tensor([30, 30, 130, 130])]

ap = calculate_ap(detections, ground_truths, iou_threshold)
print(f"Average Precision at IoU threshold {iou_threshold}: {ap}")
