##########################################
# Was used when we explored YOLO V8 NOT with resnet detection.
##########################################

import numpy as np
import matplotlib.pyplot as plt


def non_max_suppression(boxes, scores, iou_threshold=0.5):
    if len(boxes) == 0:
        return []

    # Convert the bounding boxes format from (x_center, y_center, width, height)
    # to (x_min, y_min, x_max, y_max) for easier IOU calculation
    x_min = boxes[:, 0] - boxes[:, 2] / 2
    y_min = boxes[:, 1] - boxes[:, 3] / 2
    x_max = boxes[:, 0] + boxes[:, 2] / 2
    y_max = boxes[:, 1] + boxes[:, 3] / 2

    # Stack the converted coordinates to get the format (x_min, y_min, x_max, y_max)
    boxes = np.column_stack((x_min, y_min, x_max, y_max))

    # Get the indices of the sorted scores (in descending order)
    indices = np.argsort(scores)[::-1]

    keep = []  # List to keep the indices of the final bounding boxes
    while len(indices) > 0:
        i = indices[0]  # Index of the current box with the highest score
        keep.append(i)  # Keep this box

        # Get the coordinates of the current box
        box_i = boxes[i]

        # Get the coordinates of the remaining boxes
        boxes_remaining = boxes[indices[1:]]

        # Calculate the Intersection over Union (IoU) between the current box and the remaining boxes
        inter_x_min = np.maximum(box_i[0], boxes_remaining[:, 0])
        inter_y_min = np.maximum(box_i[1], boxes_remaining[:, 1])
        inter_x_max = np.minimum(box_i[2], boxes_remaining[:, 2])
        inter_y_max = np.minimum(box_i[3], boxes_remaining[:, 3])
        inter_area = np.maximum(0, inter_x_max - inter_x_min) * np.maximum(0, inter_y_max - inter_y_min)

        box_i_area = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])
        boxes_remaining_area = (boxes_remaining[:, 2] - boxes_remaining[:, 0]) * (
                    boxes_remaining[:, 3] - boxes_remaining[:, 1])

        union_area = box_i_area + boxes_remaining_area - inter_area

        iou = inter_area / (union_area + 1e-9)  # Adding a small epsilon to avoid division by zero

        # Get the indices of the remaining boxes that have IoU less than the threshold
        # Offset by 1 as indices[0] is the current box
        indices = indices[1:][iou < iou_threshold]

    return keep  # Return the indices of the kept bounding boxes


def filter_onnx_preds(preds, min_conf=0.2, filter_non_person=True):
    filtered_preds = preds[:, preds[4:].max(axis=0) > min_conf]
    # print(f'Shape after filtering by confidence: {filtered_preds.shape}')

    preds_classes = filtered_preds[4:].argmax(axis=0)
    preds_objectness = filtered_preds[4:].max(axis=0)
    preds_bboxes = filtered_preds[:4].T

    if filter_non_person:
        preds_bboxes = preds_bboxes[preds_classes == 0]
        preds_objectness = preds_objectness[preds_classes == 0]
        preds_classes = preds_classes[preds_classes == 0]
    return preds_bboxes, preds_classes, preds_objectness


