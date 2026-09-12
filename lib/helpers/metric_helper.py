from typing import Dict, List, Sequence, Mapping
from collections import defaultdict
import logging
import json

import numpy as np
import torch


def nested_to_cpu(objects):
    """Move nested tesnors in objects to CPU if they are on GPU"""
    if isinstance(objects, torch.Tensor):
        return objects.cpu()
    elif isinstance(objects, Mapping):
        return type(objects)({k: nested_to_cpu(v) for k, v in objects.items()})
    elif isinstance(objects, (list, tuple)):
        return type(objects)([nested_to_cpu(v) for v in objects])
    elif isinstance(objects, (np.ndarray, str, int, float, bool)):
        return objects
    raise ValueError(f"Unsupported type {type(objects)}")


def calculate_3DIoU(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU of two axis-aligned 3D bboxes.
        
    Args:
        box_a (np.ndarray): 3D bbox in 6D-format (x, y, z, h, w, l)
        box_b (np.ndarray): 3D bbox in 6D-format (x, y, z, h, w, l)
    """
    max_a = box_a[0:3] + box_a[3:6] / 2
    max_b = box_b[0:3] + box_b[3:6] / 2
    min_max = np.array([max_a, max_b]).min(0)

    min_a = box_a[0:3] - box_a[3:6] / 2
    min_b = box_b[0:3] - box_b[3:6] / 2
    max_min = np.array([min_a, min_b]).max(0)
    if not ((min_max > max_min).all()):
        return 0.0

    intersection = (min_max - max_min).prod()
    vol_a = box_a[3:6].prod()
    vol_b = box_b[3:6].prod()
    union = vol_a + vol_b - intersection
    return 1.0 * intersection / union

