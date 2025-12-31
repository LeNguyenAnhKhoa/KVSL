# Các chỉ số landmark gốc từ MediaPipe Holistic
# Giống hệt với checkpoint_extraction.py và train.py
UPPER_BODY_POSE_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
FACE_LANDMARK_INDICES = []  # Không sử dụng face landmarks

# Số lượng landmarks cho mỗi phần
N_POSE_LANDMARKS = len(UPPER_BODY_POSE_INDICES)  # 23
N_FACE_LANDMARKS = len(FACE_LANDMARK_INDICES)  # 0
N_HAND_LANDMARKS = 21
N_TOTAL_LANDMARKS = N_POSE_LANDMARKS + N_FACE_LANDMARKS + N_HAND_LANDMARKS + N_HAND_LANDMARKS  # 65

# Ánh xạ từ chỉ số landmark gốc của MediaPipe Pose sang chỉ số mới trong mảng keypoints
POSE_MAP = {original_idx: new_idx for new_idx, original_idx in enumerate(UPPER_BODY_POSE_INDICES)}

# Chỉ số mới của các khớp quan trọng sau khi đã được ánh xạ
POSE_LM_LEFT_SHOULDER = POSE_MAP.get(11)
POSE_LM_RIGHT_SHOULDER = POSE_MAP.get(12)
POSE_LM_LEFT_ELBOW = POSE_MAP.get(13)
POSE_LM_RIGHT_ELBOW = POSE_MAP.get(14)
POSE_LM_LEFT_WRIST = POSE_MAP.get(15)
POSE_LM_RIGHT_WRIST = POSE_MAP.get(16)

# Chỉ số bắt đầu của các phần khác trong mảng keypoints tổng
FACE_START_IDX = N_POSE_LANDMARKS
LEFT_HAND_START_IDX = N_POSE_LANDMARKS + N_FACE_LANDMARKS
RIGHT_HAND_START_IDX = N_POSE_LANDMARKS + N_FACE_LANDMARKS + N_HAND_LANDMARKS
