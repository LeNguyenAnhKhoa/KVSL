import numpy as np
import mediapipe as mp
import random
import math
from constants import *

# --- END: Cấu hình landmarks ---

def scale_keypoints_sequence(
    keypoints_sequence, 
    scale_factor_range=(0.5, 1.5),
    num_total_landmarks=N_TOTAL_LANDMARKS,
    # Sử dụng tất cả các điểm pose đã chọn để tính trung tâm
    num_pose_landmarks_for_center=N_POSE_LANDMARKS,
    normalize_to_01=True
):
    processed_sequence = []
    if not keypoints_sequence:
        return processed_sequence
    current_scale_factor = random.uniform(scale_factor_range[0], scale_factor_range[1])
    if current_scale_factor <= 0:
        if normalize_to_01:
            temp_sequence = []
            for frame_flat in keypoints_sequence:
                if frame_flat is None:
                    temp_sequence.append(None)
                    continue
                try:
                    # frame_flat đã là (num_landmarks, 3), không cần reshape
                    if not isinstance(frame_flat, np.ndarray) or frame_flat.shape != (num_total_landmarks, 3):
                        temp_sequence.append(frame_flat.copy() if isinstance(frame_flat, np.ndarray) else frame_flat)
                        continue
                    
                    points_to_norm = frame_flat.copy()
                    valid_xy_mask_norm = np.any(points_to_norm[:, :2] != 0, axis=1)
                    if np.any(valid_xy_mask_norm):
                        x_coords = points_to_norm[valid_xy_mask_norm, 0]
                        y_coords = points_to_norm[valid_xy_mask_norm, 1]
                        min_x, max_x = np.min(x_coords), np.max(x_coords)
                        min_y, max_y = np.min(y_coords), np.max(y_coords)
                        if (max_x - min_x) > 1e-7:
                            points_to_norm[valid_xy_mask_norm, 0] = (x_coords - min_x) / (max_x - min_x)
                        elif x_coords.size > 0:
                            points_to_norm[valid_xy_mask_norm, 0] = 0.5
                        if (max_y - min_y) > 1e-7:
                            points_to_norm[valid_xy_mask_norm, 1] = (y_coords - min_y) / (max_y - min_y)
                        elif y_coords.size > 0:
                            points_to_norm[valid_xy_mask_norm, 1] = 0.5
                    # Không cần flatten, giữ nguyên shape (num_landmarks, 3)
                    temp_sequence.append(points_to_norm)
                except Exception:
                    temp_sequence.append(frame_flat.copy() if isinstance(frame_flat, np.ndarray) else frame_flat)
            return temp_sequence
        return [kp.copy() if isinstance(kp, np.ndarray) else kp for kp in keypoints_sequence]

    for frame_keypoints_flat in keypoints_sequence:
        if frame_keypoints_flat is None:
            processed_sequence.append(None)
            continue
        if not isinstance(frame_keypoints_flat, np.ndarray) or \
           frame_keypoints_flat.shape != (num_total_landmarks, 3): # <-- Sửa 1: Kiểm tra shape 2D
            processed_sequence.append(frame_keypoints_flat.copy())
            continue
        # Sửa 2: Không cần reshape, chỉ cần copy
        current_points_3d = frame_keypoints_flat.copy()

        pose_points = current_points_3d[0:N_POSE_LANDMARKS]
        # Các điểm khác để tính trung tâm (tay trái và tay phải, không có face landmarks)
        other_points_for_center = current_points_3d[N_POSE_LANDMARKS:]
        points_for_center_pose_part = pose_points[0:num_pose_landmarks_for_center]
        points_to_calculate_center_list = [points_for_center_pose_part]
        if other_points_for_center.shape[0] > 0:
            points_to_calculate_center_list.append(other_points_for_center)

        center_x, center_y = 0.0, 0.0
        can_calculate_center = False
        if points_to_calculate_center_list:
            points_to_calculate_center_concat = np.concatenate(points_to_calculate_center_list, axis=0)
            valid_center_points_mask = np.any(points_to_calculate_center_concat != 0, axis=1)
            valid_center_points = points_to_calculate_center_concat[valid_center_points_mask]
            if valid_center_points.shape[0] > 0:
                center_x = np.median(valid_center_points[:, 0])
                center_y = np.median(valid_center_points[:, 1])
                can_calculate_center = True
            else:
                all_valid_points_mask = np.any(current_points_3d != 0, axis=1)
                all_valid_points = current_points_3d[all_valid_points_mask]
                if all_valid_points.shape[0] > 0:
                    center_x = np.median(all_valid_points[:, 0])
                    center_y = np.median(all_valid_points[:, 1])
                    can_calculate_center = True

        if can_calculate_center:
            all_valid_points_mask_for_scaling = np.any(current_points_3d != 0, axis=1)
            if np.any(all_valid_points_mask_for_scaling):
                x_all_valid = current_points_3d[all_valid_points_mask_for_scaling, 0]
                y_all_valid = current_points_3d[all_valid_points_mask_for_scaling, 1]
                x_trans = x_all_valid - center_x
                y_trans = y_all_valid - center_y
                x_scaled = x_trans * current_scale_factor
                y_scaled = y_trans * current_scale_factor
                new_x_all_valid = x_scaled + center_x
                new_y_all_valid = y_scaled + center_y
                current_points_3d[all_valid_points_mask_for_scaling, 0] = new_x_all_valid
                current_points_3d[all_valid_points_mask_for_scaling, 1] = new_y_all_valid

        if normalize_to_01:
            valid_xy_mask_norm = np.any(current_points_3d[:, :2] != 0, axis=1)
            if np.any(valid_xy_mask_norm):
                x_coords = current_points_3d[valid_xy_mask_norm, 0]
                y_coords = current_points_3d[valid_xy_mask_norm, 1]
                min_x, max_x = np.min(x_coords), np.max(x_coords)
                min_y, max_y = np.min(y_coords), np.max(y_coords)
                if (max_x - min_x) > 1e-7:
                    current_points_3d[valid_xy_mask_norm, 0] = (x_coords - min_x) / (max_x - min_x)
                elif x_coords.size > 0:
                    current_points_3d[valid_xy_mask_norm, 0] = 0.5
                if (max_y - min_y) > 1e-7:
                    current_points_3d[valid_xy_mask_norm, 1] = (y_coords - min_y) / (max_y - min_y)
                elif y_coords.size > 0:
                    current_points_3d[valid_xy_mask_norm, 1] = 0.5

        # Sửa 3: Bỏ .flatten()
        if np.isnan(current_points_3d).any() or np.isinf(current_points_3d).any():
            processed_sequence.append(frame_keypoints_flat.copy())
        else:
            processed_sequence.append(current_points_3d)

    return processed_sequence

def rotate_keypoints_sequence(
    keypoints_sequence,
    angle_degrees_range=(-15.0, 15.0),
    num_total_landmarks=N_TOTAL_LANDMARKS,
    num_pose_landmarks_for_center=N_POSE_LANDMARKS
):
    rotated_sequence = []
    if not keypoints_sequence:
        return rotated_sequence
    angle_deg = random.uniform(angle_degrees_range[0], angle_degrees_range[1])
    angle_rad = math.radians(angle_deg)
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)

    for frame_keypoints_flat in keypoints_sequence:
        if frame_keypoints_flat is None:
            rotated_sequence.append(None)
            continue
        if not isinstance(frame_keypoints_flat, np.ndarray) or \
           frame_keypoints_flat.shape != (num_total_landmarks, 3): # <-- Sửa 1
            rotated_sequence.append(frame_keypoints_flat.copy())
            continue

        # Sửa 2: Bỏ reshape
        all_points = frame_keypoints_flat.copy()

        pose_points = all_points[0:N_POSE_LANDMARKS]
        # Các điểm khác để tính trung tâm (tay trái và tay phải, không có face landmarks)
        other_points_for_center = all_points[N_POSE_LANDMARKS:]
        points_for_center_pose_part = pose_points[0:num_pose_landmarks_for_center]
        points_to_calculate_center_list = [points_for_center_pose_part]
        if other_points_for_center.shape[0] > 0:
            points_to_calculate_center_list.append(other_points_for_center)

        if not points_to_calculate_center_list:
            rotated_sequence.append(frame_keypoints_flat.copy())
            continue

        points_to_calculate_center = np.concatenate(points_to_calculate_center_list, axis=0)
        valid_center_points_mask = np.any(points_to_calculate_center != 0, axis=1)
        valid_center_points = points_to_calculate_center[valid_center_points_mask]
        center_x, center_y = 0.0, 0.0
        can_calculate_center = False

        if valid_center_points.shape[0] > 0:
            center_x = np.median(valid_center_points[:, 0])
            center_y = np.median(valid_center_points[:, 1])
            can_calculate_center = True
        else:
            all_valid_points_mask = np.any(all_points != 0, axis=1)
            all_valid_points = all_points[all_valid_points_mask]
            if all_valid_points.shape[0] > 0:
                center_x = np.median(all_valid_points[:, 0])
                center_y = np.median(all_valid_points[:, 1])
                can_calculate_center = True

        if not can_calculate_center:
            rotated_sequence.append(frame_keypoints_flat.copy())
            continue

        rotated_all_points = all_points.copy()
        all_valid_points_mask_for_rotation = np.any(all_points != 0, axis=1)
        x_original_valid = all_points[all_valid_points_mask_for_rotation, 0]
        y_original_valid = all_points[all_valid_points_mask_for_rotation, 1]
        x_translated = x_original_valid - center_x
        y_translated = y_original_valid - center_y
        x_rotated = x_translated * cos_angle - y_translated * sin_angle
        y_rotated = x_translated * sin_angle + y_translated * cos_angle
        new_x_all_valid = x_rotated + center_x
        new_y_all_valid = y_rotated + center_y
        rotated_all_points[all_valid_points_mask_for_rotation, 0] = new_x_all_valid
        rotated_all_points[all_valid_points_mask_for_rotation, 1] = new_y_all_valid
        # Sửa 3: Bỏ .flatten()
        if np.isnan(rotated_all_points).any() or np.isinf(rotated_all_points).any():
            rotated_sequence.append(frame_keypoints_flat.copy())
        else:
            rotated_sequence.append(rotated_all_points)

    return rotated_sequence

def add_gaussian_noise(keypoints_sequence, mean=0.0, std=0.05):
    noisy_sequence = []
    for frame_kps in keypoints_sequence:
        if frame_kps is None:
            noisy_sequence.append(None)
            continue
        # Đảm bảo frame_kps là numpy array với shape (num_landmarks, 3)
        if not isinstance(frame_kps, np.ndarray):
            noisy_sequence.append(frame_kps)
            continue
            
        noise = np.random.normal(mean, std, frame_kps.shape)
        # Chỉ thêm nhiễu vào các keypoint đã được phát hiện (khác 0)
        mask = frame_kps != 0
        noisy_frame = frame_kps + (noise * mask)
        noisy_sequence.append(noisy_frame)
    return noisy_sequence

def temporal_padding(
    keypoints_sequence,
    padding_ratio_range=(0.05, 0.25)
):
    """
    Thêm padding (khung hình tĩnh) vào đầu và/hoặc cuối của chuỗi keypoints.
    Điều này mô phỏng hành động xảy ra ở các thời điểm khác nhau
    trong một cửa sổ thời gian dài hơn.
    """
    padded_sequence = []
    if not keypoints_sequence:
        return padded_sequence

    # Lọc ra các frame hợp lệ (không phải None)
    valid_frames = [kp for kp in keypoints_sequence if kp is not None]

    # Nếu không có frame hợp lệ, trả về chuỗi gốc (có thể là list các None)
    if not valid_frames:
        return [kp.copy() if isinstance(kp, np.ndarray) else kp for kp in keypoints_sequence]

    original_len = len(valid_frames)
    
    # Lấy frame đầu tiên và cuối cùng để làm padding
    first_frame = valid_frames[0]
    last_frame = valid_frames[-1]

    # Quyết định tổng số frame padding cần thêm, dựa trên tỷ lệ
    current_padding_ratio = random.uniform(padding_ratio_range[0], padding_ratio_range[1])
    total_padding_frames = int(round(original_len * current_padding_ratio))

    # Nếu không cần thêm padding, trả về bản sao của chuỗi gốc
    if total_padding_frames <= 0:
        return [kp.copy() if isinstance(kp, np.ndarray) else kp for kp in keypoints_sequence]

    # Phân chia ngẫu nhiên số frame padding cho phần đầu và phần cuối
    pad_start_count = random.randint(0, total_padding_frames)
    pad_end_count = total_padding_frames - pad_start_count

    # 1. Thêm padding vào đầu chuỗi (sử dụng bản sao của frame đầu tiên)
    for _ in range(pad_start_count):
        padded_sequence.append(first_frame.copy())

    # 2. Thêm các frame hành động thực tế (tạo bản sao)
    for frame in valid_frames:
        padded_sequence.append(frame.copy())

    # 3. Thêm padding vào cuối chuỗi (sử dụng bản sao của frame cuối cùng)
    for _ in range(pad_end_count):
        padded_sequence.append(last_frame.copy())

    return padded_sequence

def keypoint_dropout(keypoints_sequence, dropout_prob=0.15):
    dropped_sequence = []
    num_landmarks = N_TOTAL_LANDMARKS
    for frame_kps in keypoints_sequence:
        if frame_kps is None:
            dropped_sequence.append(None)
            continue

        new_frame = frame_kps.copy()
        for i in range(num_landmarks):
            if random.random() < dropout_prob:
                new_frame[i, :] = 0.0
        dropped_sequence.append(new_frame)
    return dropped_sequence

def translate_keypoints_sequence(
    keypoints_sequence,
    translate_x_range=(-0.3, 0.3),
    translate_y_range=(-0.3, 0.3),
    clip_to_01: bool = True,
    num_total_landmarks=N_TOTAL_LANDMARKS,
):
    translated_sequence = []
    if not keypoints_sequence:
        return translated_sequence
    dx = random.uniform(translate_x_range[0], translate_x_range[1])
    dy = random.uniform(translate_y_range[0], translate_y_range[1])

    for frame_keypoints_flat in keypoints_sequence:
        if frame_keypoints_flat is None:
            translated_sequence.append(None)
            continue
        if not isinstance(frame_keypoints_flat, np.ndarray) or \
           frame_keypoints_flat.shape != (num_total_landmarks, 3): # <-- Sửa 1
            translated_sequence.append(frame_keypoints_flat.copy())
            continue

        # Sửa 2: Bỏ reshape
        all_points = frame_keypoints_flat.copy()

        translated_all_points = all_points.copy()
        valid_points_mask = np.any(all_points != 0, axis=1)
        translated_all_points[valid_points_mask, 0] += dx
        translated_all_points[valid_points_mask, 1] += dy

        if clip_to_01:
            translated_all_points[valid_points_mask, 0] = np.clip(translated_all_points[valid_points_mask, 0], 0.0, 1.0)
            translated_all_points[valid_points_mask, 1] = np.clip(translated_all_points[valid_points_mask, 1], 0.0, 1.0)

        # Sửa 3: Bỏ .flatten()
        if np.isnan(translated_all_points).any() or np.isinf(translated_all_points).any():
            translated_sequence.append(frame_keypoints_flat.copy())
        else:
            translated_sequence.append(translated_all_points)

    return translated_sequence

def time_stretch_keypoints_sequence(
    keypoints_sequence,
    speed_factor_range=(0.5, 1.5),
):
    perturbed_sequence = []
    if not keypoints_sequence or all(kp is None for kp in keypoints_sequence):
        return keypoints_sequence

    valid_frames = [kp for kp in keypoints_sequence if kp is not None]
    if not valid_frames:
        return keypoints_sequence

    original_num_valid_frames = len(valid_frames)
    current_speed_factor = random.uniform(speed_factor_range[0], speed_factor_range[1])
    if current_speed_factor <= 0:
        return [kp.copy() if isinstance(kp, np.ndarray) else kp for kp in keypoints_sequence]
    if current_speed_factor == 1.0:
        return [kp.copy() if isinstance(kp, np.ndarray) else kp for kp in keypoints_sequence]

    num_new_frames = int(round(original_num_valid_frames / current_speed_factor))
    if num_new_frames == 0:
        if original_num_valid_frames > 0:
            perturbed_sequence.append(valid_frames[0].copy() if valid_frames[0] is not None else None)
        return perturbed_sequence

    original_indices = np.linspace(0, original_num_valid_frames - 1, num_new_frames)
    resampled_indices = np.round(original_indices).astype(int)
    resampled_indices = np.clip(resampled_indices, 0, original_num_valid_frames - 1)

    for res_idx in resampled_indices:
        perturbed_sequence.append(valid_frames[res_idx].copy())

    return perturbed_sequence

# Các chỉ số này đã được định nghĩa ở đầu file và đã được ánh xạ
# POSE_LM_LEFT_SHOULDER
# POSE_LM_RIGHT_SHOULDER
# POSE_LM_LEFT_ELBOW
# POSE_LM_RIGHT_ELBOW
# POSE_LM_LEFT_WRIST
# POSE_LM_RIGHT_WRIST

def solve_2_link_ik_2d_v2(
    p_shoulder_xy: np.ndarray,
    p_wrist_target_xy: np.ndarray,
    len_upper_arm: float,
    len_forearm: float,
    original_elbow_xy=None,
    original_wrist_xy=None,
    prefer_original_bend: bool = True
):
    d = np.linalg.norm(p_wrist_target_xy - p_shoulder_xy)
    l1 = max(1e-5, len_upper_arm)
    l2 = max(1e-5, len_forearm)

    if d > l1 + l2 - 1e-5:
        if d < 1e-9:
            return p_shoulder_xy + np.array([l1, 0]) if original_elbow_xy is None else original_elbow_xy.copy()
        vec_sw = (p_wrist_target_xy - p_shoulder_xy) / d
        return p_shoulder_xy + vec_sw * l1

    if d < abs(l1 - l2) + 1e-5:
        if original_elbow_xy is not None:
            return original_elbow_xy.copy()
        if d < 1e-9:
            return p_shoulder_xy + np.array([l1, 0])
        vec_sw = (p_wrist_target_xy - p_shoulder_xy) / d
        return p_shoulder_xy + vec_sw * l1

    if d < 1e-9: d = 1e-9
    a = (l1**2 - l2**2 + d**2) / (2 * d)
    h_squared = l1**2 - a**2
    if h_squared < -1e-9:
        vec_sw = (p_wrist_target_xy - p_shoulder_xy) / d
        return p_shoulder_xy + vec_sw * l1
    h = np.sqrt(max(0, h_squared))
    p2_x = p_shoulder_xy[0] + a * (p_wrist_target_xy[0] - p_shoulder_xy[0]) / d
    p2_y = p_shoulder_xy[1] + a * (p_wrist_target_xy[1] - p_shoulder_xy[1]) / d
    perp_vec_x = -(p_wrist_target_xy[1] - p_shoulder_xy[1]) / d
    perp_vec_y = (p_wrist_target_xy[0] - p_shoulder_xy[0]) / d
    elbow_sol1_xy = np.array([p2_x + h * perp_vec_x, p2_y + h * perp_vec_y])
    elbow_sol2_xy = np.array([p2_x - h * perp_vec_x, p2_y - h * perp_vec_y])

    if not prefer_original_bend or original_elbow_xy is None or original_wrist_xy is None:
        if original_elbow_xy is not None:
            dist1 = np.linalg.norm(elbow_sol1_xy - original_elbow_xy)
            dist2 = np.linalg.norm(elbow_sol2_xy - original_elbow_xy)
            return elbow_sol1_xy if dist1 <= dist2 else elbow_sol2_xy
        return elbow_sol1_xy

    vec_sw_orig = original_wrist_xy - p_shoulder_xy
    if np.linalg.norm(vec_sw_orig) < 1e-5:
        dist1 = np.linalg.norm(elbow_sol1_xy - original_elbow_xy)
        dist2 = np.linalg.norm(elbow_sol2_xy - original_elbow_xy)
        return elbow_sol1_xy if dist1 <= dist2 else elbow_sol2_xy

    original_side = (original_wrist_xy[0] - p_shoulder_xy[0]) * (original_elbow_xy[1] - p_shoulder_xy[1]) - \
                    (original_wrist_xy[1] - p_shoulder_xy[1]) * (original_elbow_xy[0] - p_shoulder_xy[0])
    side1 = (p_wrist_target_xy[0] - p_shoulder_xy[0]) * (elbow_sol1_xy[1] - p_shoulder_xy[1]) - \
            (p_wrist_target_xy[1] - p_shoulder_xy[1]) * (elbow_sol1_xy[0] - p_shoulder_xy[0])
    side2 = (p_wrist_target_xy[0] - p_shoulder_xy[0]) * (elbow_sol2_xy[1] - p_shoulder_xy[1]) - \
            (p_wrist_target_xy[1] - p_shoulder_xy[1]) * (elbow_sol2_xy[0] - p_shoulder_xy[0])

    if abs(original_side) < 1e-3:
        dist1 = np.linalg.norm(elbow_sol1_xy - original_elbow_xy)
        dist2 = np.linalg.norm(elbow_sol2_xy - original_elbow_xy)
        return elbow_sol1_xy if dist1 <= dist2 else elbow_sol2_xy

    if np.sign(side1) == np.sign(original_side):
        return elbow_sol1_xy
    elif np.sign(side2) == np.sign(original_side):
        return elbow_sol2_xy
    else:
        dist1 = np.linalg.norm(elbow_sol1_xy - original_elbow_xy)
        dist2 = np.linalg.norm(elbow_sol2_xy - original_elbow_xy)
        return elbow_sol1_xy if dist1 <= dist2 else elbow_sol2_xy

def inter_hand_distance(
    keypoints_sequence,
    total_dx_change_range=(-0.1, 0.1),
    overall_dy_shift_range=(-0.1, 0.1),
    clip_to_01: bool = True,
    num_total_landmarks: int = N_TOTAL_LANDMARKS,
):
    augmented_sequence = []
    if not keypoints_sequence: return augmented_sequence
    current_total_dx_change = random.uniform(total_dx_change_range[0], total_dx_change_range[1])
    current_overall_dy_shift = random.uniform(overall_dy_shift_range[0], overall_dy_shift_range[1])

    for frame_keypoints_flat in keypoints_sequence:
        if frame_keypoints_flat is None:
            augmented_sequence.append(None)
            continue
        if not isinstance(frame_keypoints_flat, np.ndarray) or \
           frame_keypoints_flat.shape != (num_total_landmarks, 3): # <-- Sửa 1
            augmented_sequence.append(frame_keypoints_flat.copy())
            continue
        # Sửa 2: Bỏ reshape
        all_points_orig = frame_keypoints_flat.copy()

        augmented_points = all_points_orig.copy()

        # Kiểm tra xem các chỉ số khớp cần thiết có tồn tại không
        required_indices = [
            POSE_LM_LEFT_SHOULDER, POSE_LM_LEFT_ELBOW, POSE_LM_LEFT_WRIST,
            POSE_LM_RIGHT_SHOULDER, POSE_LM_RIGHT_ELBOW, POSE_LM_RIGHT_WRIST
        ]
        if any(idx is None for idx in required_indices):
            # Nếu một trong các khớp quan trọng không có trong POSE_MAP, bỏ qua augmentation này
            augmented_sequence.append(frame_keypoints_flat.copy())
            continue

        s_l_orig_xy = all_points_orig[POSE_LM_LEFT_SHOULDER, 0:2].copy()
        e_l_orig_xy = all_points_orig[POSE_LM_LEFT_ELBOW, 0:2].copy()
        w_l_orig_xy = all_points_orig[POSE_LM_LEFT_WRIST, 0:2].copy()
        s_r_orig_xy = all_points_orig[POSE_LM_RIGHT_SHOULDER, 0:2].copy()
        e_r_orig_xy = all_points_orig[POSE_LM_RIGHT_ELBOW, 0:2].copy()
        w_r_orig_xy = all_points_orig[POSE_LM_RIGHT_WRIST, 0:2].copy()
        left_arm_key_points_valid = np.all(s_l_orig_xy != 0) and np.all(e_l_orig_xy != 0) and np.all(w_l_orig_xy != 0)
        right_arm_key_points_valid = np.all(s_r_orig_xy != 0) and np.all(e_r_orig_xy != 0) and np.all(w_r_orig_xy != 0)

        if np.all(w_l_orig_xy != 0) and np.all(w_r_orig_xy != 0):
            current_mid_wrists_x = (w_l_orig_xy[0] + w_r_orig_xy[0]) / 2
            x_left = min(w_l_orig_xy[0], w_r_orig_xy[0])
            x_right = max(w_l_orig_xy[0], w_r_orig_xy[0])
            current_dist_wrists_x = x_right - x_left
            target_dist_wrists_x = current_dist_wrists_x + current_total_dx_change
            if target_dist_wrists_x < 0.01: target_dist_wrists_x = 0.01
            if w_l_orig_xy[0] <= w_r_orig_xy[0]:
                w_l_target_x = current_mid_wrists_x - target_dist_wrists_x / 2
                w_r_target_x = current_mid_wrists_x + target_dist_wrists_x / 2
            else:
                w_r_target_x = current_mid_wrists_x - target_dist_wrists_x / 2
                w_l_target_x = current_mid_wrists_x + target_dist_wrists_x / 2
        else:
            w_l_target_x = w_l_orig_xy[0]
            w_r_target_x = w_r_orig_xy[0]

        if left_arm_key_points_valid:
            len_l_upper = np.linalg.norm(e_l_orig_xy - s_l_orig_xy)
            len_l_forearm = np.linalg.norm(w_l_orig_xy - e_l_orig_xy)
            w_l_target_xy_for_ik = np.array([w_l_target_x, w_l_orig_xy[1]])
            e_l_new_xy = solve_2_link_ik_2d_v2(s_l_orig_xy, w_l_target_xy_for_ik, len_l_upper, len_l_forearm, e_l_orig_xy, w_l_orig_xy)
            if e_l_new_xy is not None:
                dx_wrist_l = w_l_target_xy_for_ik[0] - w_l_orig_xy[0]
                dy_wrist_l = w_l_target_xy_for_ik[1] - w_l_orig_xy[1]
                augmented_points[POSE_LM_LEFT_ELBOW, 0:2] = e_l_new_xy
                augmented_points[POSE_LM_LEFT_WRIST, 0:2] = w_l_target_xy_for_ik
                idx_lh_start = LEFT_HAND_START_IDX
                idx_lh_end = idx_lh_start + N_HAND_LANDMARKS
                left_hand_kps_part = augmented_points[idx_lh_start:idx_lh_end]
                left_hand_valid_mask = np.any(left_hand_kps_part != 0, axis=1)
                if np.any(left_hand_valid_mask):
                    left_hand_kps_part[left_hand_valid_mask, 0] += dx_wrist_l
                    left_hand_kps_part[left_hand_valid_mask, 1] += dy_wrist_l
                augmented_points[idx_lh_start:idx_lh_end] = left_hand_kps_part

        if right_arm_key_points_valid:
            len_r_upper = np.linalg.norm(e_r_orig_xy - s_r_orig_xy)
            len_r_forearm = np.linalg.norm(w_r_orig_xy - e_r_orig_xy)
            w_r_target_xy_for_ik = np.array([w_r_target_x, w_r_orig_xy[1]])
            e_r_new_xy = solve_2_link_ik_2d_v2(s_r_orig_xy, w_r_target_xy_for_ik, len_r_upper, len_r_forearm, e_r_orig_xy, w_r_orig_xy)
            if e_r_new_xy is not None:
                dx_wrist_r = w_r_target_xy_for_ik[0] - w_r_orig_xy[0]
                dy_wrist_r = w_r_target_xy_for_ik[1] - w_r_orig_xy[1]
                augmented_points[POSE_LM_RIGHT_ELBOW, 0:2] = e_r_new_xy
                augmented_points[POSE_LM_RIGHT_WRIST, 0:2] = w_r_target_xy_for_ik
                idx_rh_start = RIGHT_HAND_START_IDX
                idx_rh_end = idx_rh_start + N_HAND_LANDMARKS
                right_hand_kps_part = augmented_points[idx_rh_start:idx_rh_end]
                right_hand_valid_mask = np.any(right_hand_kps_part != 0, axis=1)
                if np.any(right_hand_valid_mask):
                    right_hand_kps_part[right_hand_valid_mask, 0] += dx_wrist_r
                    right_hand_kps_part[right_hand_valid_mask, 1] += dy_wrist_r
                augmented_points[idx_rh_start:idx_rh_end] = right_hand_kps_part

        # Khởi tạo danh sách các chỉ số cần xử lý (cánh tay và bàn tay)
        arm_and_hand_indices = [
            POSE_LM_LEFT_WRIST, POSE_LM_LEFT_ELBOW,
            POSE_LM_RIGHT_WRIST, POSE_LM_RIGHT_ELBOW
        ]
        # Thêm tất cả các landmark của cả hai bàn tay
        arm_and_hand_indices.extend(list(range(LEFT_HAND_START_IDX, RIGHT_HAND_START_IDX + N_HAND_LANDMARKS)))
        
        # Lọc bỏ các giá trị None nếu có
        unique_arm_hand_indices = sorted(list(set(idx for idx in arm_and_hand_indices if idx is not None)))

        if abs(current_overall_dy_shift) > 1e-5:
            for idx in unique_arm_hand_indices:
                # Xác định xem landmark có thuộc cánh tay hoặc bàn tay hợp lệ không
                is_left_arm_part = (idx == POSE_LM_LEFT_WRIST or idx == POSE_LM_LEFT_ELBOW)
                is_right_arm_part = (idx == POSE_LM_RIGHT_WRIST or idx == POSE_LM_RIGHT_ELBOW)
                is_left_hand_part = (LEFT_HAND_START_IDX <= idx < RIGHT_HAND_START_IDX)
                is_right_hand_part = (RIGHT_HAND_START_IDX <= idx < num_total_landmarks)

                should_shift_y = (is_left_arm_part and left_arm_key_points_valid) or \
                                 (is_right_arm_part and right_arm_key_points_valid) or \
                                 (is_left_hand_part and left_arm_key_points_valid) or \
                                 (is_right_hand_part and right_arm_key_points_valid)

                if should_shift_y and idx < len(augmented_points) and np.any(augmented_points[idx, 0:2] != 0):
                    augmented_points[idx, 1] += current_overall_dy_shift

        if clip_to_01:
            # Chỉ clip các landmark đã bị thay đổi (cánh tay và bàn tay)
            indices_to_clip = unique_arm_hand_indices
            for idx in indices_to_clip:
                if idx < len(augmented_points) and np.any(augmented_points[idx, 0:2] != 0):
                    augmented_points[idx, 0] = np.clip(augmented_points[idx, 0], 0.0, 1.0)
                    augmented_points[idx, 1] = np.clip(augmented_points[idx, 1], 0.0, 1.0)

        # Sửa 3: Bỏ .flatten()
        if np.isnan(augmented_points).any() or np.isinf(augmented_points).any():
            augmented_sequence.append(frame_keypoints_flat.copy())
        else:
            augmented_sequence.append(augmented_points)

    return augmented_sequence