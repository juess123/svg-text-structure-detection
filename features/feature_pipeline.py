from core.parser import  parse_commands,command_stats
from core.sampling import sample_path
from core.geometry import compute_total_length,compute_bbox
from features.feature_utils import (
    compute_fill_ratio,
    compute_relative_bbox_area,
    compute_direction_change_ratio,
    compute_turning_density,
    compute_small_segment_ratio,
    compute_curve_ratio,
    compute_avg_commands_per_subpath,
    compute_compactness,
    compute_mean_curvature,
    compute_command_density,
    compute_subpath_density,
    compute_point_density,
    compute_avg_segment_length,
    compute_segment_length_std,
    compute_direction_variance,

    compute_command_entropy,
    compute_curvature_std,
    compute_normalized_length
)

def extract_features_from_raw_d(raw_d, svg_area=None):

    commands = parse_commands(raw_d)
    stats = command_stats(commands)

    points = sample_path(commands)
    length = compute_total_length(points)
    bbox = compute_bbox(points)
       # 计算 bbox
    bbox = compute_bbox(points)


    relative_bbox_area = compute_relative_bbox_area(bbox,svg_area)
    
    fill_ratio = compute_fill_ratio(points, bbox)
    direction_change_ratio = compute_direction_change_ratio(points)
    turning_density = compute_turning_density(points)
    small_segment_ratio = compute_small_segment_ratio(points)

    # 原有特征
    curve_ratio = compute_curve_ratio(stats)
    avg_cmd_sub = compute_avg_commands_per_subpath(stats)
    compactness = compute_compactness(length, bbox)
    mean_curvature = compute_mean_curvature(points)

    command_density = compute_command_density(stats, length)
    subpath_density = compute_subpath_density(stats, bbox)
    point_density = compute_point_density(points, length)
    avg_seg_len = compute_avg_segment_length(points)
    seg_len_std = compute_segment_length_std(points)
    dir_var = compute_direction_variance(points)

    # 新增特征
  
    command_entropy = compute_command_entropy(commands)
    curvature_std = compute_curvature_std(points)
    normalized_length = compute_normalized_length(length, bbox)
    
    return {

        "direction_change_ratio": direction_change_ratio,
        "turning_density": turning_density,
        "small_segment_ratio": small_segment_ratio,
        "relative_bbox_area":relative_bbox_area,
        "fill_ratio": fill_ratio,
        # 结构比例类
        "curve_ratio": curve_ratio,
        "avg_cmd_per_subpath": avg_cmd_sub,
        "compactness": compactness,
        "normalized_length": normalized_length,

        # 曲率类
        "mean_curvature": mean_curvature,
        "curvature_std": curvature_std,

        # 密度类
        "command_density": command_density,
        "subpath_density": subpath_density,
        "point_density": point_density,

        # 线段统计
        "avg_segment_length": avg_seg_len,
        "segment_length_std": seg_len_std,

        # 方向统计
        "direction_variance": dir_var,

        # 拓扑统计
        "command_entropy": command_entropy
    }