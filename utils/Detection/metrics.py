import numpy as np


def calculate_iou(box1, box2):
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])

    # Calculate intersection area
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate areas of the two bounding boxes
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]

    # Calculate IoU
    iou = intersection_area / (area1 + area2 - intersection_area)

    return iou


def calculate_precision_recall(pred_boxes, true_boxes, iou_threshold):
    """
    x: The x-coordinate of the top-left corner of the bounding box.
    y: The y-coordinate of the top-left corner of the bounding box.
    width: The width of the bounding box.
    height: The height of the bounding box.
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    identified_object_indices = set()

    for pred_box in pred_boxes:
        matched = False
        for true_box_idx, true_box in enumerate(true_boxes):
            iou = calculate_iou(pred_box, true_box)
            if iou >= iou_threshold:
                true_positives += 1
                matched = True
                identified_object_indices.add(true_box_idx)
                break
        if not matched:
            false_positives += 1

    false_negatives = len(true_boxes) - len(identified_object_indices)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 0

    return precision, recall


def calc_f1(precision, recall):
    return (2*precision*recall)/(precision+recall) if (precision+recall)!=0 else 0

