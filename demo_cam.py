import cv2
import numpy as np
import torch
import mediapipe as mp
from model.pose_transformer import Action_LSTM

# Mediapipe 설정
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True)
# 모델 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Action_LSTM().to(device)
model_path = 'checkpoint/epoch_3.pth'  # 모델 체크포인트 파일 경로
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 액션 라벨
action_names = ['start', 'stop', 'stand']

# 비디오 캡처 시작
cap = cv2.VideoCapture(4)  # 0은 기본 웹캠

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("비디오 읽기 실패")
        break
    
    # Mediapipe로 포즈 추출
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    
    # 포즈 랜드마크 처리 및 모델 입력 준비
    if results.pose_landmarks:
        landmarks = np.array([[lm.x, lm.y] for lm in results.pose_landmarks.landmark]).flatten()
        landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32).to(device).unsqueeze(0)
        
        # 모델 예측
        with torch.no_grad():
            output = model(landmarks_tensor)
            predicted_action = torch.argmax(output, dim=1)
            label = action_names[predicted_action.item()]
            
            # 라벨 표시
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # 처리된 프레임 표시
    cv2.imshow("Action Recognition", frame)
    
    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()