import mediapipe as mp
import numpy as np
import random 
from tqdm import tqdm
import time
import cv2
import os
import logging
import requests

from transformers import AdamW
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch

from torch.utils.data.dataloader import default_collate
from model.pose_transformer import Action_LSTM
from common.utils import load_data_from_npz, save_as_npz, skeleton_to_image, save_model_epoch
from common.generator import *
from common.dataset import ActionDataset, split_dataset
from configs.args import args_parse

args = args_parse().parse()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args, train_loader, model, optimizer, epoch):
    return step(args, 'train', train_loader, model, optimizer, epoch)

def test(args, test_loader, model, epoch):
    with torch.no_grad():
        return step(args, 'test', test_loader, model, epoch)

def step(args, split, dataloader, model, optimizer=None, epoch=None):
    
    criterion = nn.CrossEntropyLoss()
    if split == 'train':
        model.train()
        total_loss = 0
        
        for i, (inputs, labels) in enumerate(tqdm(dataloader)):
            # inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch: {epoch}, Loss: {avg_loss:.4f}')
        return avg_loss
    
    else:
        model.eval()
        total_loss = 0
        correct_predictions = 0
        with torch.no_grad():  # 그라디언트 계산 비활성화
            for inputs, labels in tqdm(dataloader):
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                predictions = outputs.argmax(dim=1)    # 가장 높은 로짓을 가진 인덱스를 예측 값으로 선택
                correct_predictions += (predictions == labels).sum().item()
            avg_loss = total_loss / len(dataloader)
            accuracy = correct_predictions / len(dataloader.dataset)
            print(f'Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
            return avg_loss, accuracy
        

def main():
    args.manualSeed = 777
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    
    abs_path = os.path.abspath(__file__)
    root_path = os.path.dirname(abs_path)
    
    if args.train:
        logtime = time.strftime('_%m-%d_%H:%M')
        log_dir = 'ARFormer'
        log_path = root_path + 'log/' + log_dir + logtime
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        
        opts = dict((name, getattr(args, name)) for name in dir(args) if not name.startswith('_'))
        file_name = os.path.join(log_path, 'args.txt')
        with open(file_name, 'wt') as args_file:
            args_file.write('==> Args: \n')
            for k, v in sorted(opts.items()):
                args_file.write(' %s: %s\n' % (str(k), str(v)))
            args_file.write('==> Args:\n')
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
                    filename=os.path.join(log_path, 'train.log'), level=logging.INFO)
        

    data_path = root_path  + '/data/'
    train_keypoint_npz_path = os.path.join(data_path, args.train_dataset + '.npz')
    test_keypoint_npz_path = os.path.join(data_path, args.test_dataset + '.npz')
    keypoint_dict = extract_landmarks(data_path)
    
    # video2npz 생성
    # save_as_npz(keypoint_dict, keypoint_npz_path)
    train_list, test_list = load_data_from_npz(train_keypoint_npz_path, test_keypoint_npz_path)
    train_dataset = ActionDataset(train_list)
    test_dataset = ActionDataset(test_list)
    
    def collate_fn(batch):
        inputs, labels = zip(*batch)
        inputs = torch.stack([item for item in inputs])
        labels = torch.tensor(labels)
        return inputs, labels

    if args.train:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                shuffle=True, pin_memory=True, collate_fn=collate_fn)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
    #                         shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                            shuffle=False, pin_memory=True)
    
    model = Action_LSTM().to(device)

    model_dict = model.state_dict()
    if args.previous_dir != '':
        model_paths = sorted(glob.glob(os.path.join(args.previous_dir, '*.pth')))

        for path in model_paths:
            if path.split('/')[-1].startswith('model'):
                model_path = path

        pre_dict = torch.load(model_path)

        model_dict = model.state_dict()
        state_dict = {k: v for k, v in pre_dict.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(1, args.nepoch):
        if args.train:
            loss = train(args, train_dataloader, model, optimizer, epoch)
        
        _, accuracy = test(args, test_loader, model, epoch)
        if args.train:
            if args.previous_best_threshold < accuracy:
                print(f'Test Accuracy: {accuracy:.4f}')
                args.previous_best_threshold = accuracy
                if os.path.exists(args.checkpoint):
                    os.makedirs(args.checkpoint)
                save_model_epoch(args.checkpoint, epoch, model)
            
            
if __name__ == '__main__':
    main()
    