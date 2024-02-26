import mediapipe as mp
import os
import cv2
import random
import numpy as np

skeleton = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
draw_line = [[11, 13], [13, 15], [12, 14], [14, 16], [23, 25], [25, 27],
             [24, 26], [26, 28], [11, 12], [11, 23], [23, 24], [12, 24]]
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True)

def extract_landmarks(video_path):
    video_name_lst = os.listdir(video_path)
    keypoint_dict = {}
    for video in video_name_lst:
        # 사용하는 video 이름으로 변경 & label 매핑
        if video.startswith('standing'): label = 0
        elif video.startswith('starting'): label = 1
        elif video.startswith('stopping'): label = 2
        else: continue
        # else: label = 2
        
        full_video_path = os.path.join(video_path, video)
        
        cap = cv2.VideoCapture(full_video_path)
        landmarks_list = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                skeleton_lst = []
                for idx, lm in enumerate(results.pose_landmarks.landmark):
                    if idx in skeleton:
                        skeleton_lst.append((lm.x, lm.y))
                # landmarks = [(lm.x, lm.y) for lm in results.pose_landmarks.landmark]
                # for idx, kp in enumerate(landmarks):
                #     if idx in skeleton:
                        # skeleton_lst.append(skeleton)
                landmarks_list.append(skeleton_lst) 
        cap.release()
        if landmarks_list:
            basename, ext = os.path.splitext(video)
            keypoint_dict[basename] = {'label': label, 'keypoints': landmarks_list}
            
    return keypoint_dict

def flip_landmarks(landmarks):
    flipped = [(1-x, y) for x, y in landmarks]  # 가로 방향 뒤집기
    return flipped

def rotate_landmarks(landmarks, angle):
    theta = np.radians(angle)
    cos, sin = np.cos(theta), np.sin(theta)
    rotated = [(x*cos - y*sin, x*sin + y*cos) for x, y in landmarks]  # 회전
    return rotated

def transform(landmarks):
    if random.random() > 0.5:
        landmarks = flip_landmarks(landmarks)
    if random.random() > 0.5:
        angle = random.randint(-10, 10)  # -10도에서 10도 사이에서 랜덤하게 회전
        landmarks = rotate_landmarks(landmarks, angle)
    return landmarks