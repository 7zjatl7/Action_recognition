import cv2
import sys
import os
import os.path as osp
import numpy as np
import torch
import mediapipe as mp
from model.pose_transformer import Action_LSTM
from configs.args import args_parse
from common.vis import video_vis

skeleton = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
draw_line = [[11, 13], [13, 15], [12, 14], [14, 16], [23, 25], [25, 27],
             [24, 26], [26, 28], [11, 12], [11, 23], [23, 24], [12, 24]]

path = os.path.abspath(__file__)
root_path = os.path.dirname(path)
video_path = os.listdir(root_path)

args = args_parse().parse()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess_landmarks(landmarks_list):
    # 랜드마크 리스트를 텐서로 변환하는 전처리 과정
    tensor_input = torch.tensor(landmarks_list, dtype=torch.float).to(device)
    return tensor_input


def main():
    
    model_path = 'checkpoint/epoch_%d.pth' % int(args.test_epoch)
    assert osp.exists(model_path), 'Cannot find model at ' + model_path
    
    model = Action_LSTM().to(device)
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt)
    model.eval()
    # img_path = 'start.JPG'
    img_path = 'stop.JPG'
    # img_path = 'stand.JPG'
    
    frame = cv2.imread(img_path)
    # mediapipe 가져오기
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True)
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    landmarks_list = []
    if results.pose_landmarks:
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            if idx in skeleton:
                landmarks_list.append(lm.x)
                landmarks_list.append(lm.y)
    
    keypoints_list = []
    
    landmarks_tensor = preprocess_landmarks([landmarks_list])
    
    # for kp in input_tensor:
    #     kp_flat = np.concatenate(kp).ravel()
    #     keypoints_list.append(kp_flat)
        
    print(landmarks_tensor.shape)
    
    with torch.no_grad():
        result = model(landmarks_tensor)
        predicted_action = torch.argmax(result, dim=1)

    action_names = ['stand', 'start', 'stop']  # 예시 액션 이름
    predicted_action_name = action_names[predicted_action.item()]
    
    print(f"Predicted Action: {predicted_action_name}")
    
    

if __name__ == '__main__':
    main()