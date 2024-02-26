import cv2

def camera_vis(action):
    cap = cv2.VideoCapture(6)
        
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))	# 영상의 넓이(가로) 프레임
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (frameWidth, frameHeight)
    
    frameRate = 33
    
    if not cap.isOpened():
        print('Camera not connected')
        return
    
    while True:
        ret, frame = cap.read()
        
        if not(ret):	# 프레임정보를 정상적으로 읽지 못하면
            break  # while문을 빠져나가기
        
        cv2.putText(frame, action, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Webcam live for action recognition', frame)	# 프레임 보여주기
        key = cv2.waitKey(frameRate)  # frameRate msec동안 한 프레임을 보여준다
        
        # 'q' 키를 누르면 루프에서 빠져나옵니다.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()