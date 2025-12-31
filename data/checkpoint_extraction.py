import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import os
import mediapipe as mp
import pandas as pd
from tqdm import tqdm
import json
from scipy.interpolate import interp1d
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from augment_function import inter_hand_distance, scale_keypoints_sequence, rotate_keypoints_sequence, translate_keypoints_sequence, time_stretch_keypoints_sequence, add_gaussian_noise, keypoint_dropout, temporal_padding
from constants import *

mp_holistic = mp.solutions.holistic

# --- START: CÁC THAY ĐỔI CHÍNH ---

# Các hằng số về landmarks giờ đây được import từ constants.py
# N_POSE_LANDMARKS, N_FACE_LANDMARKS, N_HAND_LANDMARKS,
# UPPER_BODY_POSE_INDICES, FACE_LANDMARK_INDICES đã có sẵn

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    pose_kps = np.zeros((N_POSE_LANDMARKS, 3))
    face_kps = np.zeros((N_FACE_LANDMARKS, 3))
    left_hand_kps = np.zeros((N_HAND_LANDMARKS, 3))
    right_hand_kps = np.zeros((N_HAND_LANDMARKS, 3))

    if results and results.pose_landmarks:
        for i, idx in enumerate(UPPER_BODY_POSE_INDICES):
            if idx < len(results.pose_landmarks.landmark):
                res = results.pose_landmarks.landmark[idx]
                pose_kps[i] = [res.x, res.y, res.z]

    if results and results.face_landmarks:
        for i, idx in enumerate(FACE_LANDMARK_INDICES):
            if idx < len(results.face_landmarks.landmark):
                res = results.face_landmarks.landmark[idx]
                face_kps[i] = [res.x, res.y, res.z]

    if results and results.left_hand_landmarks:
        left_hand_kps = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark])

    if results and results.right_hand_landmarks:
        right_hand_kps = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark])

    keypoints = np.concatenate([pose_kps, face_kps, left_hand_kps, right_hand_kps])
    return keypoints

# --- END: CÁC THAY ĐỔI CHÍNH (Giữ nguyên các phần còn lại) ---

def interpolate_keypoints(keypoints_sequence, target_len=60):
    if not keypoints_sequence:
        return None
    
    # Lọc bỏ các frame None
    valid_frames = [frame for frame in keypoints_sequence if frame is not None]
    
    num_frames = len(valid_frames)
    if num_frames == 0:
        return None

    indices = np.linspace(0, num_frames - 1, target_len).astype(int)
    downsampled_sequence = [valid_frames[i] for i in indices]
    
    # Kiểm tra tất cả frames có cùng shape không
    shapes = [frame.shape for frame in downsampled_sequence if isinstance(frame, np.ndarray)]
    if len(shapes) == 0:
        return None
    
    unique_shapes = set(shapes)
    if len(unique_shapes) > 1:
        print(f"Cảnh báo: Phát hiện nhiều shape khác nhau trong sequence: {unique_shapes}")
        return None
    
    expected_shape = shapes[0]
    if len(expected_shape) != 2 or expected_shape[1] != 3:
        print(f"Cảnh báo: Shape không hợp lệ {expected_shape}, mong đợi (num_landmarks, 3)")
        return None
    
    # Đảm bảo tất cả frame có cùng shape trước khi stack
    try:
        result = np.array(downsampled_sequence)
        # Kiểm tra shape phải là (target_len, num_landmarks, 3)
        if result.ndim == 3 and result.shape[0] == target_len and result.shape[2] == 3:
            return result
        else:
            print(f"Cảnh báo: Result shape không đúng: {result.shape}, mong đợi ({target_len}, num_landmarks, 3)")
            return None
    except Exception as e:
        print(f"Lỗi khi tạo array từ sequence: {e}")
        return None

def sequence_frames(video_path, holistic):
    sequence_frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % step != 0:
            continue

        try:
            image, results = mediapipe_detection(frame, holistic)
            keypoints = extract_keypoints(results)
            if keypoints is not None:
                sequence_frames.append(keypoints)
        except:
            continue

    cap.release()
    return sequence_frames

def generate_augmented_samples(original_sequence, augmentation_functions, num_samples_to_generate, max_augs_per_sample=3):
    generated_samples = []
    if not original_sequence or not augmentation_functions:
        return generated_samples

    num_available_augs = len(augmentation_functions)

    for i in range(num_samples_to_generate):
        current_sequence = [kp.copy() if isinstance(kp, np.ndarray) else kp for kp in original_sequence]
        num_augs_to_apply = random.randint(1, min(max_augs_per_sample, num_available_augs))
        selected_aug_funcs_indices = random.sample(range(num_available_augs), num_augs_to_apply)
        selected_aug_funcs = [augmentation_functions[idx] for idx in selected_aug_funcs_indices]
        random.shuffle(selected_aug_funcs)

        for aug_func in selected_aug_funcs:
            current_sequence = aug_func(current_sequence)
            if not current_sequence or all(frame is None for frame in current_sequence):
                break

        if not current_sequence or all(frame is None for frame in current_sequence):
            continue

        generated_samples.append(current_sequence)

    return generated_samples

augmentations = [
    scale_keypoints_sequence,
    rotate_keypoints_sequence,
    translate_keypoints_sequence,
    time_stretch_keypoints_sequence,
    inter_hand_distance,
    add_gaussian_noise,
    keypoint_dropout
]

def process_video(row): # Xóa tham số 'holistic'
    video_id = row['ID']
    action = row['LABEL']
    video_file = row['VIDEO']
    
    video_path = os.path.join(video_folder, video_file)
    if not os.path.exists(video_path):
        print(f"Video không tồn tại: {video_path}")
        return None
    
    id_path = os.path.join(DATA_PATH, str(video_id))
    os.makedirs(id_path, exist_ok=True)
    
    # TẠO MỚI holistic model Ở ĐÂY, bên trong hàm của luồng
    # Sử dụng 'with' để đảm bảo model được giải phóng đúng cách
    with mp_holistic.Holistic(
        model_complexity=1, 
        min_detection_confidence=0.7, 
        min_tracking_confidence=0.7, 
        smooth_landmarks=True
    ) as holistic:
        frame_lists = sequence_frames(video_path, holistic)

    # Các phần còn lại của hàm giữ nguyên
    if not frame_lists: # Thêm kiểm tra nếu video không trích xuất được frame nào
        print(f"Không trích xuất được frame từ video ID {video_id}")
        return None

    # Kiểm tra shape của frame_lists
    if len(frame_lists) > 0:
        first_frame_shape = frame_lists[0].shape if isinstance(frame_lists[0], np.ndarray) else None
        print(f"Video ID {video_id}: {len(frame_lists)} frames, shape mỗi frame: {first_frame_shape}")

    augmenteds = generate_augmented_samples(frame_lists, augmentations, 1500, 8)
    augmenteds.append(frame_lists)
    
    saved_count = 0
    for idx, aug in enumerate(augmenteds):
        seq = interpolate_keypoints(aug)
        if seq is not None:
            file_path = os.path.join(id_path, f'{idx}.npz')
            np.savez(file_path, sequence=seq, label=label_map[action])
            saved_count += 1
    
    print(f"Video ID {video_id}: Đã lưu {saved_count}/{len(augmenteds)} sequences")
    return video_id

DATA_PATH = 'Data'
label_file = 'Dataset/data.csv'
video_folder = 'Dataset/Videos'

df = pd.read_csv(label_file)

os.makedirs(DATA_PATH, exist_ok=True)

unique_labels = sorted(df['LABEL'].unique())
label_map = {action: idx for idx, action in enumerate(unique_labels)}

label_map_path = os.path.join(DATA_PATH, 'label_map.json')
with open(label_map_path, 'w', encoding='utf-8') as f:
    json.dump(label_map, f, ensure_ascii=False, indent=4)

dataset_label_map_path = os.path.join('Dataset', 'do_vat.json')
with open(dataset_label_map_path, 'w', encoding='utf-8') as f:
    json.dump(label_map, f, ensure_ascii=False, indent=4)

workers = min(3, os.cpu_count())

with ThreadPoolExecutor(max_workers=workers) as executor:
    # BỎ dòng tạo holistic_models
    futures = []
    
    # Lặp và submit tác vụ mà KHÔNG cần truyền holistic
    for _, row in df.iterrows():
        future = executor.submit(process_video, row)
        futures.append(future)
    
    for future in tqdm(as_completed(futures), total=len(futures)):
        try: # Thêm try-except để bắt lỗi nếu có
            future.result()
        except Exception as e:
            print(f"An error occurred in a thread: {e}")