from core.svg_path_ops import parse_commands, command_stats, sample_path
import numpy as np
from core.geometry import compute_total_length, compute_bbox
from features.feature_utils import (
    compute_fill_ratio,
    compute_direction_change_ratio,
    compute_small_segment_ratio,
    compute_subpath_density,
    compute_point_density,
    compute_avg_segment_length,
    compute_segment_length_std,
    compute_direction_variance,
    compute_curvature_std,
    compute_sharp_turn_count,
    compute_closed_subpath_ratio,
    compute_command_type_ratios,
)

def extract_features_from_raw_d(raw_d, svg_area=None):
    commands = parse_commands(raw_d)
    stats = command_stats(commands)

    points = sample_path(commands)
    length = compute_total_length(points)
    bbox = compute_bbox(points)

    # 统一 bbox 格式
    bbox_w = max(bbox["bbox_width"], 1e-6)
    bbox_h = max(bbox["bbox_height"], 1e-6)
    bbox_area = bbox_w * bbox_h

    fill_ratio = compute_fill_ratio(points, bbox)
    direction_change_ratio = compute_direction_change_ratio(points)
    small_segment_ratio = compute_small_segment_ratio(points)

    subpath_density = compute_subpath_density(stats, bbox)
    point_density = compute_point_density(points, length)

    avg_seg_len = compute_avg_segment_length(points) / max(length, 1e-6)
    seg_len_std = compute_segment_length_std(points) / max(length, 1e-6)
    dir_var = compute_direction_variance(points) / (np.pi ** 2)
    curvature_std = compute_curvature_std(points)

    # 新增特征
    aspect_ratio = bbox_w / bbox_h
    log_aspect_ratio = np.log(aspect_ratio + 1e-6)

    length_per_bbox_area = length / max(bbox_area, 1e-6)
    compactness2 = (length * length) / max(bbox_area, 1e-6)

    sharp_turn_count = compute_sharp_turn_count(points)
    sharp_turn_density = sharp_turn_count / max(length, 1e-6)

    subpath_count = stats.get("num_subpaths", 0)
    closed_subpath_ratio = compute_closed_subpath_ratio(commands)

    line_ratio, curve_ratio, move_ratio, close_ratio = compute_command_type_ratios(commands)

    return {
        "direction_change_ratio": direction_change_ratio,
        "small_segment_ratio": small_segment_ratio,
        "fill_ratio": fill_ratio,
        "subpath_density": subpath_density,
        "point_density": point_density,
        "avg_segment_length": avg_seg_len,
        "segment_length_std": seg_len_std,
        "direction_variance": dir_var,
        "curvature_std": curvature_std,

        "log_aspect_ratio": log_aspect_ratio,
        "length_per_bbox_area": length_per_bbox_area,
        "compactness2": compactness2,
        "sharp_turn_density": sharp_turn_density,
        "subpath_count": subpath_count,
        "closed_subpath_ratio": closed_subpath_ratio,
        "line_ratio": line_ratio,
        "curve_ratio": curve_ratio,
        "move_ratio": move_ratio,
        "close_ratio": close_ratio,
    }