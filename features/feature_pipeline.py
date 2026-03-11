from core.parser import  parse_commands,command_stats
from core.sampling import sample_path
import numpy as np   
from core.geometry import compute_total_length,compute_bbox
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
)

def extract_features_from_raw_d(raw_d, svg_area=None):

    commands = parse_commands(raw_d)
    stats = command_stats(commands)

    points = sample_path(commands)
    length = compute_total_length(points)
    bbox = compute_bbox(points)
    fill_ratio = compute_fill_ratio(points, bbox)
    direction_change_ratio = compute_direction_change_ratio(points)
    small_segment_ratio = compute_small_segment_ratio(points)
    # 原有特征
    subpath_density = compute_subpath_density(stats, bbox)
    point_density = compute_point_density(points, length)
    avg_seg_len = compute_avg_segment_length(points)/length 
    seg_len_std = compute_segment_length_std(points)/length
    dir_var = compute_direction_variance(points) / (np.pi ** 2)
    # 新增特征
    curvature_std = compute_curvature_std(points)
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
}