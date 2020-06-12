import torch
from model import *
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from torchvision import datasets


if __name__ == "__main__":
    trans = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_data = datasets.ImageFolder('./samples/', trans)
    print(train_data.class_to_idx)

    testloader = DataLoader(train_data, batch_size=8)
    model = Resnet18()
    model.load_state_dict(torch.load("../resnet18_office.pth", map_location='cpu'))
    model.train(False)

    for input, label in testloader:
        res = model(input)
        res = torch.argmax(res, 1)
        print(res)
        print(label)
