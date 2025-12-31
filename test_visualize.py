"""
Simple hardcoded test for keypoint visualization
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

def test_keypoint_visualization():
    # Hardcoded test data
    data_folder = "data/Data"
    class_id = 347
    sample_id = 0
    
    # Load data
    file_path = os.path.join(data_folder, str(class_id), f"{sample_id}.npz")
    data = np.load(file_path)
    sequence = data['sequence']  # Shape: (60, 195) - flattened
    label = data['label']
    
    print(f"Loaded: Class {class_id}, Sample {sample_id}")
    print(f"Sequence shape: {sequence.shape}")
    print(f"Label: {label}")
    
    # Reshape to (60, 65, 3) - 195 = 65 * 3
    sequence = sequence.reshape(60, 65, 3)
    print(f"Reshaped to: {sequence.shape}")
    
    # Parse keypoints (25 pose + 20 left_hand + 20 right_hand = 65)
    def parse_keypoints(keypoints_3d):
        pose_kps = keypoints_3d[:25]  # First 25 landmarks
        left_hand_kps = keypoints_3d[25:45]  # Next 20 landmarks
        right_hand_kps = keypoints_3d[45:65]  # Last 20 landmarks
        return pose_kps, left_hand_kps, right_hand_kps
    
    # Create animation
    fig, ax = plt.subplots(figsize=(10, 8))
    
    def animate(frame_idx):
        ax.clear()
        
        if frame_idx >= len(sequence):
            return
        
        # Get keypoints for current frame
        keypoints = sequence[frame_idx]  # Shape: (65, 3)
        pose_kps, left_hand_kps, right_hand_kps = parse_keypoints(keypoints)
        
        # Plot pose landmarks (red)
        valid_pose = pose_kps[:, 2] > 0
        if np.any(valid_pose):
            ax.scatter(pose_kps[valid_pose, 0], pose_kps[valid_pose, 1], 
                      c='red', s=50, alpha=0.7, label='Pose')
        
        # Plot left hand landmarks (blue)
        valid_left = left_hand_kps[:, 2] > 0
        if np.any(valid_left):
            ax.scatter(left_hand_kps[valid_left, 0], left_hand_kps[valid_left, 1], 
                      c='blue', s=40, alpha=0.7, label='Left Hand')
        
        # Plot right hand landmarks (green)
        valid_right = right_hand_kps[:, 2] > 0
        if np.any(valid_right):
            ax.scatter(right_hand_kps[valid_right, 0], right_hand_kps[valid_right, 1], 
                      c='green', s=40, alpha=0.7, label='Right Hand')
        
        # Draw simple connections for pose
        for i in range(0, len(pose_kps)-1, 2):
            if (pose_kps[i, 2] > 0 and pose_kps[i+1, 2] > 0):
                ax.plot([pose_kps[i, 0], pose_kps[i+1, 0]], 
                       [pose_kps[i, 1], pose_kps[i+1, 1]], 
                       'r-', alpha=0.5, linewidth=1)
        
        # Draw simple connections for hands
        for i in range(0, len(left_hand_kps)-1, 2):
            if (left_hand_kps[i, 2] > 0 and left_hand_kps[i+1, 2] > 0):
                ax.plot([left_hand_kps[i, 0], left_hand_kps[i+1, 0]], 
                       [left_hand_kps[i, 1], left_hand_kps[i+1, 1]], 
                       'b-', alpha=0.5, linewidth=1)
        
        for i in range(0, len(right_hand_kps)-1, 2):
            if (right_hand_kps[i, 2] > 0 and right_hand_kps[i+1, 2] > 0):
                ax.plot([right_hand_kps[i, 0], right_hand_kps[i+1, 0]], 
                       [right_hand_kps[i, 1], right_hand_kps[i+1, 1]], 
                       'g-', alpha=0.5, linewidth=1)
        
        # Set labels and title
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Frame {frame_idx+1}/{len(sequence)} - Class {label}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()  # Invert Y axis to match image coordinates
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=len(sequence), 
                                 interval=150, repeat=True)
    
    plt.tight_layout()
    plt.show()
    
    return anim

if __name__ == "__main__":
    test_keypoint_visualization()
