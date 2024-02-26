import numpy as np
import torch


skeleton = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# connections = [[11, 13], [13, 15], [12, 14], [14, 16], [23, 25], [25, 27],
#              [24, 26], [26, 28], [11, 12], [11, 23], [23, 24], [12, 24]]

connections = [[0, 2], [2, 4], [1, 3], [3, 5], [6, 8], [8, 10],
             [7, 9], [9, 11], [0, 1], [0, 6], [6, 7], [1, 7]]


def skeleton_to_image(data, img_size=(224, 224)):
    """
    2D 스켈레톤 데이터를 이미지로 변환합니다.
    skeleton: 각 관절의 (x, y) 좌표 리스트
    img_size: 출력 이미지의 크기
    """
    keypoint_label_dict = {}
    
    for i in range(len(data['keypoints'])):
        keypoints = data['keypoints'][i]
        xy_pairs = [(keypoints[i], keypoints[i + 1]) for j in range(0, len(keypoints), 2)]
        keypoint_label_dict[int(data['labels'][i])] = xy_pairs
        # image = np.zeros(img_size + (3,))  # RGB 이미지
        
        # plt.figure(figsize=(img_size[0] / 100, img_size[1] / 100), dpi=100)
        # plt.axis('off')
        
        # (x, y) 좌표 점 그리기
        # x_coords = [x for x, y in xy_pairs]
        # y_coords = [y for x, y in xy_pairs]
        # plt.scatter(x_coords, y_coords, s=10)
        
        # for start, end in connections:
        #     plt.plot([xy_pairs[start][0], xy_pairs[end][0]], [xy_pairs[start][1], xy_pairs[end][1]], 'ro-')

        # plt.scatter(*zip(*data['keypoints'][i]), s=10)  # 관절 위치 그리기
        # 여기에 추가적으로 관절을 연결하는 선을 그릴 수 있습니다.
        # plt.xlim(0, img_size[0])
        # plt.ylim(0, img_size[1])
        # plt.gca().invert_yaxis()  # y축 반전
        # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        # image_normalized = image.astype(np.float32) / 255.0
        # # 이미지로 변환
        # plt.savefig("temp.png")
        # plt.close()
        # image = Image.open("temp.png")
        # keypoint_label_dict[data['labels'][i]] = {image}
        
    return xy_pairs


# def load_data_from_npz(npz_file):
    # data = np.load(npz_file, allow_pickle=True)
    # return data

def load_data_from_npz(train_data, test_data):
    train_data = np.load(train_data, allow_pickle=True)
    test_data = np.load(test_data, allow_pickle=True)
    train_keypoints = train_data['keypoints']
    train_labels = train_data['labels']
    
    test_keypoints = test_data['keypoints']
    test_labels = test_data['labels']
    
    return [train_keypoints, train_labels], [test_keypoints, test_labels]


def save_as_npz(data_dict, save_path):
    # 저장할 레이블과 키포인트 리스트를 초기화합니다.
    labels_list = []
    keypoints_list = []

    # data_dict를 순회하며 레이블과 키포인트를 추출합니다.
    for video, info in data_dict.items():
        label = info['label']
        keypoints = info['keypoints']
        
        # 각 키포인트 리스트를 순회하며 평탄화하고, 레이블과 함께 저장합니다.
        for kp in keypoints:
            # 평탄화(flatten)된 키포인트를 리스트에 추가합니다.
            kp_flat = np.concatenate(kp).ravel()
            keypoints_list.append(kp_flat)
            labels_list.append(label)
    
    # 리스트를 NumPy 배열로 변환합니다.
    labels_array = np.array(labels_list, dtype=np.int32)
    keypoints_array = np.array(keypoints_list, dtype=np.float32)
    
    # .npz 파일로 저장합니다.
    np.savez_compressed(save_path, labels=labels_array, keypoints=keypoints_array)

def save_model_epoch(save_dir, epoch, model):
    torch.save(model.state_dict(), '%s/epoch_%d.pth' % (save_dir, epoch))