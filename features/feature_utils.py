import numpy as np
from collections import Counter
import math
def compute_polygon_area(points):
    """
    使用 Shoelace formula 计算多边形面积
    """
    if len(points) < 3:
        return 0.0

    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])

    return 0.5 * abs(np.dot(x, np.roll(y, -1)) -
                     np.dot(y, np.roll(x, -1)))


def compute_fill_ratio(points, bbox_dict):
    """
    填充度 = path面积 / bbox面积
    """
    area = compute_polygon_area(points)

    width = bbox_dict["bbox_width"]
    height = bbox_dict["bbox_height"]

    bbox_area = width * height

    return area / (bbox_area + 1e-6)





def compute_curve_ratio(stats):
    if stats["num_commands"] == 0:
        return 0.0
    return stats["num_curve_commands"] / stats["num_commands"]


def compute_avg_commands_per_subpath(stats):
    if stats["num_subpaths"] == 0:
        return 0.0
    return stats["num_commands"] / stats["num_subpaths"]


def compute_compactness(total_length, bbox):
    perimeter = 2 * (bbox["bbox_width"] + bbox["bbox_height"])
    if perimeter == 0:
        return 0.0
    return total_length / perimeter



def compute_mean_curvature(points):
    """
    计算平均转角（近似曲率）
    """

    if len(points) < 3:
        return 0.0

    angles = []

    for i in range(1, len(points)-1):
        v1 = points[i] - points[i-1]
        v2 = points[i+1] - points[i]

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            continue

        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        angle = np.arccos(cos_angle)
        angles.append(angle)

    if len(angles) == 0:
        return 0.0

    return float(np.mean(angles))



def compute_command_density(stats, total_length):
    if total_length == 0:
        return 0.0
    return stats["num_commands"] / total_length


def compute_subpath_density(stats, bbox):
    area = bbox["bbox_width"] * bbox["bbox_height"]
    if area == 0:
        return 0.0
    return stats["num_subpaths"] / area


def compute_point_density(points, total_length):
    if total_length == 0:
        return 0.0
    return len(points) / total_length


def compute_avg_segment_length(points):
    if len(points) < 2:
        return 0.0
    diffs = np.diff(points, axis=0)
    lengths = np.linalg.norm(diffs, axis=1)
    return float(np.mean(lengths))


def compute_segment_length_std(points):
    if len(points) < 2:
        return 0.0
    diffs = np.diff(points, axis=0)
    lengths = np.linalg.norm(diffs, axis=1)
    return float(np.std(lengths))


def compute_direction_variance(points):
    if len(points) < 2:
        return 0.0

    diffs = np.diff(points, axis=0)
    angles = np.arctan2(diffs[:, 1], diffs[:, 0])

    return float(np.var(angles))





def compute_command_entropy(commands):
    types = [cmd.lower() for cmd,_ in commands]
    total = len(types)
    if total == 0:
        return 0.0

    counter = Counter(types)
    entropy = -sum((v/total) * math.log(v/total) for v in counter.values())
    return entropy

def compute_curvature_std(points):
    if len(points) < 3:
        return 0.0

    angles = []

    for i in range(1, len(points)-1):
        v1 = points[i] - points[i-1]
        v2 = points[i+1] - points[i]

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            continue

        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        angle = np.arccos(cos_angle)
        angles.append(angle)

    if not angles:
        return 0.0

    return float(np.std(angles))

def compute_normalized_length(total_length, bbox):
    scale = bbox["bbox_width"] + bbox["bbox_height"]
    if scale == 0:
        return 0.0
    return total_length / scale

def compute_relative_bbox_area(bbox, svg_area):
    """
    计算 path 的 bbox 面积占整个 SVG 面积的比例
    """

    if bbox is None:
        return 0.0

    path_width = bbox.get("bbox_width", 0.0)
    path_height = bbox.get("bbox_height", 0.0)

    path_bbox_area = path_width * path_height

    if svg_area is not None and svg_area > 0:
        return path_bbox_area / (svg_area + 1e-6)

    return 0.0

def compute_direction_change_ratio(points, angle_threshold=np.pi/6):
    """
    方向变化频率 = 大角度转折次数 / 总段数
    """

    if len(points) < 3:
        return 0.0

    points = np.array(points)

    # 计算方向向量
    vecs = points[1:] - points[:-1]

    # 去掉零长度段
    lengths = np.linalg.norm(vecs, axis=1)
    valid = lengths > 1e-6
    vecs = vecs[valid]

    if len(vecs) < 2:
        return 0.0

    # 计算方向角
    angles = np.arctan2(vecs[:,1], vecs[:,0])

    # 方向差
    angle_diff = np.abs(np.diff(angles))

    # wrap 到 [0, pi]
    angle_diff = np.minimum(angle_diff, 2*np.pi - angle_diff)

    # 大角度次数
    big_turns = np.sum(angle_diff > angle_threshold)

    return big_turns / len(angle_diff)

def compute_turning_density(points, angle_threshold=np.pi/6):
    if len(points) < 3:
        return 0.0

    points = np.array(points)
    vecs = points[1:] - points[:-1]
    seg_lengths = np.linalg.norm(vecs, axis=1)

    valid = seg_lengths > 1e-6
    vecs = vecs[valid]
    seg_lengths = seg_lengths[valid]

    if len(vecs) < 2:
        return 0.0

    angles = np.arctan2(vecs[:,1], vecs[:,0])
    angle_diff = np.abs(np.diff(angles))
    angle_diff = np.minimum(angle_diff, 2*np.pi - angle_diff)

    big_turns = np.sum(angle_diff > angle_threshold)
    total_length = np.sum(seg_lengths)

    return big_turns / (total_length + 1e-6)

def compute_small_segment_ratio(points, length_threshold=5.0):
    if len(points) < 2:
        return 0.0

    points = np.array(points)
    vecs = points[1:] - points[:-1]
    seg_lengths = np.linalg.norm(vecs, axis=1)

    if len(seg_lengths) == 0:
        return 0.0

    small_count = np.sum(seg_lengths < length_threshold)

    return small_count / len(seg_lengths)