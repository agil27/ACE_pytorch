#!/usr/bin/env python
# coding: utf-8

import torch
from torch.optim import Adam
import os
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from model import Resnet18
from tcav import TCAV
from model_wrapper import ModelWrapper
from mydata import MyDataset


def data_loader(base_path, class_name):
    data_transforms = transforms.Compose([
        # transforms.Resize(self.input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_dataset_train = MyDataset(base_path, class_name, transform=data_transforms)
    train_loader = DataLoader(image_dataset_train, batch_size=32)
    return train_loader


def train():
    best_weights = model.state_dict()
    best_acc = 0.0
    for epoch in range(200):
        # test phase
        total = 0
        score = 0
        with torch.no_grad():
            model.eval()
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                predicted = outputs.max(dim=1)[1]
                total += labels.size(0)
                score += predicted.eq(labels).sum().item()
        acc = score / total
        print("epoch: {}\tacc: {}".format(epoch, acc))
        if acc > best_acc:
            best_acc = acc
            best_weights = model.state_dict()

        # train phase
        model.train()
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # save model parameters
    torch.save(best_weights, 'resnet18_office.pth')


def validate(model):
    model.eval()
    weights = torch.load('resnet18_office.pth')
    print(model)
    model.load_state_dict(weights)
    model = feature_layers


if __name__ == "__main__":
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    data_transforms = transforms.Compose([
        transforms.Resize([128, 128]),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    image_dataset = datasets.ImageFolder('data/amazon', data_transforms)
    train_size = int(len(image_dataset) * 0.8)
    train_data, test_data = torch.utils.data.random_split(image_dataset, [train_size, len(image_dataset) - train_size])
    trainloader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=8)
    testloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)

    # concept_dataset = datasets.ImageFolder('data/cluster_result/bike', data_transforms)
    # concept_loader = DataLoader(concept_dataset, batch_size=128, shuffle=False, num_workers=8)
    concept_dict = {}
    concept_dict['screen'] = data_loader('data/cluster_result/phone', '02')
    concept_dict['key'] = data_loader('data/cluster_result/phone', '05')

    print(concept_dict.keys())
    model = Resnet18(output_num=31)
    model = model.to(device)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.0001)

    # train()
    validate(model)
    # print(concept_loader.dataset)
