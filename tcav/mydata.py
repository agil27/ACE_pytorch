import numpy as np
#from skimage import io
#from skimage import transform
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from PIL import Image

# step1: 定义MyDataset类， 继承Dataset, 重写抽象方法：__len()__, __getitem()__
class MyDataset(Dataset):
    def __init__(self, base_path, class_name, transform=None):
        self.base_path = base_path
        self.class_name = class_name
        self.transform = transform
        self.names_list = []
        for root, dirs, files in os.walk(base_path+'/'+class_name):  
            self.names_list = files
        self.size = len(self.names_list)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image_path = self.base_path+'/'+self.class_name+'/'+self.names_list[idx]
        filename = self.names_list[idx]
        if not os.path.isfile(image_path):
            print(image_path + ' does not exist!')
            return None
        with open(image_path, 'rb') as f:
            with Image.open(f) as image:
                image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return (image, filename)