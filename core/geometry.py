import numpy as np


def compute_total_length(points):
    """
    计算路径总长度
    points: Nx2 的 numpy 数组
    """

    # 如果点太少，长度为 0
    if len(points) < 2:
        return 0.0

    # 相邻点差分
    diffs = np.diff(points, axis=0)

    # 每段长度 = sqrt(dx^2 + dy^2)
    segment_lengths = np.linalg.norm(diffs, axis=1)

    # 总长度
    total_length = np.sum(segment_lengths)

    return float(total_length)

import numpy as np


def compute_bbox(points):
    """
    计算路径的包围盒（bounding box）

    返回：
    {
        "bbox_width": ...,
        "bbox_height": ...,
        "aspect_ratio": ...
    }
    """

    # 如果没有点
    if len(points) == 0:
        return {
            "bbox_width": 0.0,
            "bbox_height": 0.0,
            "aspect_ratio": 0.0
        }

    # 取 x 和 y 的最小最大值
    min_xy = np.min(points, axis=0)
    max_xy = np.max(points, axis=0)

    width = max_xy[0] - min_xy[0]
    height = max_xy[1] - min_xy[1]

    # 避免除 0
    if height != 0:
        aspect_ratio = width / height
    else:
        aspect_ratio = 0.0

    return {
        "bbox_width": float(width),
        "bbox_height": float(height),
        "aspect_ratio": float(aspect_ratio)
    }