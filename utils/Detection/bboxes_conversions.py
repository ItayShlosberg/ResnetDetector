import numpy as np

def xywh_to_xycwh(xywh_boxes):
    x, y, w, h = xywh_boxes[:, 0], xywh_boxes[:, 1], xywh_boxes[:, 2], xywh_boxes[:, 3]
    x_center = x + w / 2
    y_center = y + h / 2
    return np.column_stack((x_center, y_center, w, h))


def xycwh_to_xywh(xycwh_boxes):
    x_center, y_center, w, h = xycwh_boxes[:, 0], xycwh_boxes[:, 1], xycwh_boxes[:, 2], xycwh_boxes[:, 3]
    x = x_center - w / 2
    y = y_center - h / 2
    return np.column_stack((x, y, w, h))


def xywh_to_x1y1x2y2(xywh_boxes):
    x, y, w, h = xywh_boxes[:, 0], xywh_boxes[:, 1], xywh_boxes[:, 2], xywh_boxes[:, 3]
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    return np.column_stack((x1, y1, x2, y2))


def x1y1x2y2_to_xywh(x1y1x2y2_boxes):
    x1, y1, x2, y2 = x1y1x2y2_boxes[:, 0], x1y1x2y2_boxes[:, 1], x1y1x2y2_boxes[:, 2], x1y1x2y2_boxes[:, 3]
    x = x1
    y = y1
    w = x2 - x1
    h = y2 - y1
    return np.column_stack((x, y, w, h))

