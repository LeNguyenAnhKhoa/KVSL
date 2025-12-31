import numpy as np
from .constants import (
    UPPER_BODY_POSE_INDICES,
    FACE_LANDMARK_INDICES,
    N_POSE_LANDMARKS,
    N_FACE_LANDMARKS,
    N_HAND_LANDMARKS
)

def extract_keypoints_from_results(results):
    """
    Trích xuất keypoints từ MediaPipe results GIỐNG HỆT checkpoint_extraction.py:
    - 23 upper body pose landmarks (TẤT CẢ 23 pose landmarks từ index 0-22)
    - 0 face landmarks (FACE_LANDMARK_INDICES = [] nhưng vẫn tạo array để concatenate đúng format)
    - 21 left hand landmarks
    - 21 right hand landmarks
    Tổng: 65 landmarks x 3 = 195 features (23+0+21+21 = 65)
    
    THỨ TỰ CONCATENATE: pose -> face -> left_hand -> right_hand (GIỐNG checkpoint_extraction.py)
    """
    pose_kps = np.zeros((N_POSE_LANDMARKS, 3))  # (23, 3)
    face_kps = np.zeros((N_FACE_LANDMARKS, 3))  # (0, 3)
    left_hand_kps = np.zeros((N_HAND_LANDMARKS, 3))  # (21, 3)
    right_hand_kps = np.zeros((N_HAND_LANDMARKS, 3))  # (21, 3)
    
    # Trích xuất ĐÚNG 23 pose landmarks theo UPPER_BODY_POSE_INDICES
    # UPPER_BODY_POSE_INDICES = [0, 1, 2, ..., 22] - tất cả 23 pose landmarks
    if results.get('poseLandmarks'):
        pose_landmarks = results['poseLandmarks']
        if isinstance(pose_landmarks, list) and len(pose_landmarks) >= 23:
            # Lấy ĐÚNG các indices theo UPPER_BODY_POSE_INDICES
            for i, idx in enumerate(UPPER_BODY_POSE_INDICES):
                if idx < len(pose_landmarks):
                    lm = pose_landmarks[idx]
                    if isinstance(lm, dict):
                        pose_kps[i] = [lm.get('x', 0), lm.get('y', 0), lm.get('z', 0)]
                    else:
                        pose_kps[i] = [getattr(lm, 'x', 0), getattr(lm, 'y', 0), getattr(lm, 'z', 0)]
    
    # Face landmarks - KHÔNG trích xuất vì FACE_LANDMARK_INDICES = []
    # face_kps đã là np.zeros((0, 3))
    
    # Trích xuất TẤT CẢ 21 left hand landmarks
    if results.get('leftHandLandmarks'):
        left_hand_landmarks = results['leftHandLandmarks']
        if isinstance(left_hand_landmarks, list) and len(left_hand_landmarks) >= 21:
            for i in range(21):
                lm = left_hand_landmarks[i]
                if isinstance(lm, dict):
                    left_hand_kps[i] = [lm.get('x', 0), lm.get('y', 0), lm.get('z', 0)]
                else:
                    left_hand_kps[i] = [getattr(lm, 'x', 0), getattr(lm, 'y', 0), getattr(lm, 'z', 0)]
    
    # Trích xuất TẤT CẢ 21 right hand landmarks
    if results.get('rightHandLandmarks'):
        right_hand_landmarks = results['rightHandLandmarks']
        if isinstance(right_hand_landmarks, list) and len(right_hand_landmarks) >= 21:
            for i in range(21):
                lm = right_hand_landmarks[i]
                if isinstance(lm, dict):
                    right_hand_kps[i] = [lm.get('x', 0), lm.get('y', 0), lm.get('z', 0)]
                else:
                    right_hand_kps[i] = [getattr(lm, 'x', 0), getattr(lm, 'y', 0), getattr(lm, 'z', 0)]
    
    # QUAN TRỌNG: Concatenate theo THỨ TỰ GIỐNG checkpoint_extraction.py
    # THỨ TỰ: pose -> face -> left_hand -> right_hand
    keypoints = np.concatenate([pose_kps, face_kps, left_hand_kps, right_hand_kps])
    # Shape: (23 + 0 + 21 + 21, 3) = (65, 3)
    return keypoints

def interpolate_sequence(sequence, target_len=60):
    if len(sequence) == 0:
        return None
    
    num_frames = len(sequence)
    indices = np.linspace(0, num_frames - 1, target_len).astype(int)
    downsampled = [sequence[i] for i in indices]
    return np.array(downsampled)