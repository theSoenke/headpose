import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision
from data import Pose300WLP
from hopenet import Hopenet
from torch.utils.data import DataLoader
from torchvision import transforms


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', dest='batch_size', default=64, type=int)
    parser.add_argument('--lr', dest='lr', default=1e-5, type=float)
    parser.add_argument('--data_dir', default='', type=str, required=True)
    parser.add_argument('--alpha', help='Regression loss coefficient.', default=2, type=float)
    parser.add_argument('--checkpoint', default='', type=str)

    args = parser.parse_args()
    return args


def get_ignored_params(model):
    # Generator function that yields ignored params.
    b = [model.conv1, model.bn1, model.fc_finetune]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param


def get_non_ignored_params(model):
    # Generator function that yields params that will be optimized.
    b = [model.layer1, model.layer2, model.layer3, model.layer4]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param


def get_fc_params(model):
    # Generator function that yields fc layer params.
    b = [model.fc_yaw, model.fc_pitch, model.fc_roll]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param


def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)


if __name__ == '__main__':
    args = parse_args()

    epochs = args.epochs
    batch_size = args.batch_size

    model = Hopenet()
    if args.checkpoint == '':
        load_filtered_state_dict(model, model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth'))
    else:
        saved_state_dict = torch.load(args.checkpoint)
        model.load_state_dict(saved_state_dict)

    transformations = transforms.Compose([
        transforms.Resize(240),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    os.makedirs('checkpoints', exist_ok=True)
    files = os.listdir(args.data_dir)
    files = [file[:-4] for file in files]
    pose_dataset = Pose300WLP(args.data_dir, files, transformations)
    train_loader = torch.utils.data.DataLoader(
        dataset=pose_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    reg_criterion = nn.MSELoss().to(device)
    alpha = args.alpha

    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(device)

    optimizer = torch.optim.Adam([
        {'params': get_ignored_params(model), 'lr': 0},
        {'params': get_non_ignored_params(model), 'lr': args.lr},
        {'params': get_fc_params(model), 'lr': args.lr * 5}
    ], lr=args.lr)

    grad_seq = [torch.ones(1).to(device) for _ in range(3)]
    for epoch in range(epochs):
        for i, (images, labels, cont_labels, name) in enumerate(train_loader):
            images = images.to(device)

            label_yaw = labels[:, 0].to(device)
            label_pitch = labels[:, 1].to(device)
            label_roll = labels[:, 2].to(device)

            label_yaw_cont = cont_labels[:, 0].to(device)
            label_pitch_cont = cont_labels[:, 1].to(device)
            label_roll_cont = cont_labels[:, 2].to(device)

            # Forward pass
            yaw, pitch, roll = model(images)

            loss_yaw = criterion(yaw, label_yaw)
            loss_pitch = criterion(pitch, label_pitch)
            loss_roll = criterion(roll, label_roll)

            # MSE loss
            yaw_predicted = F.softmax(yaw, dim=1)
            pitch_predicted = F.softmax(pitch, dim=1)
            roll_predicted = F.softmax(roll, dim=1)

            yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 99
            pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 99
            roll_predicted = torch.sum(roll_predicted * idx_tensor, 1) * 3 - 99

            loss_reg_yaw = reg_criterion(yaw_predicted, label_yaw_cont)
            loss_reg_pitch = reg_criterion(pitch_predicted, label_pitch_cont)
            loss_reg_roll = reg_criterion(roll_predicted, label_roll_cont)

            # Total loss
            loss_yaw += alpha * loss_reg_yaw
            loss_pitch += alpha * loss_reg_pitch
            loss_roll += alpha * loss_reg_roll

            loss_seq = [loss_yaw, loss_pitch, loss_roll]
            torch.autograd.backward(loss_seq, grad_seq)
            optimizer.step()
            optimizer.zero_grad()

            if (i+1) % 20 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Losses: Yaw %.4f, Pitch %.4f, Roll %.4f'
                      % (
                          epoch+1,
                          epochs, i+1,
                          len(pose_dataset)//batch_size,
                          loss_yaw.item(),
                          loss_pitch.item(),
                          loss_roll.item())
                      )

        torch.save(model.state_dict(), 'checkpoints/epoch_' + str(epoch+1) + '.pkl')
