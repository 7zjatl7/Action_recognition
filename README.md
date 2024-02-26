# Project Name

## Introduction

해당 프로젝트는 스터디용도로 구현하였습니다. 
비디오 내에서 사람의 동작을 인식하고 분류하는 Action Recognition 프로젝트입니다.
Mediapipe에서 얻은 2D Keypoints를 입력으로 사용하였습니다. 


## Dependencies

List all the dependencies required to run your project. Specify versions if necessary. For example:

- python >= 3.8
- Pytorch >= 1.13.1+cu117
- NumPy >= 1.14.4
- opencv-python >= 4.9.0
- mediapipe >= 0.10.10

ubuntu 20.04에서 실험하였습니다.

## Data
data 폴더를 새로 생성하여, 해당 위치에 비디오를 넣어주세요.
비디오의 이름을 분류할 Action의 이름으로 변경해주세요.
generator.py의 파일에서 확장자 제외하고, 비디오의 이름으로 label에 맞춰서 수정해주세요.
main.py에서 mediapipe를 통해 2d keypoints를 생성하고 싶다면, keypoint_dict = extract_landmarks(data_path)의 주석을 풀어주세요.
추출한 2d keypoints를 npz로 만들고 싶다면, save_as_npz(keypoint_dict, keypoint_npz_path)의 주석을 풀어주세요.
Train 데이터셋이랑 Test 데이터셋 각각 생성해주세요.
생성한 .npz 파일이름을 기반으로 --train_dataset train_npz_file_name, --test_dataset test_npz_file_name 으로 인자를 넣어주세요.

## Model

해당 Pre-trained 모델은 다음 링크에서 받아보실 수 있습니다. https://drive.google.com/file/d/1TKEqReZTlLORH5xRcDhhiJKWKPPdU0W3/view?usp=sharing
checkpoint 폴더안에 모델을 넣어주세요. 폴더가 없다면 새로 생성해주세요.


## Quick Demo

빠르게 데모를 실행해보고 싶다면, 다음과 같이 실행해주세요.

```bash
python demo_img.py
```
```bash
python demo_cam.py
```

## Train
```bash
python main.py --train 1

