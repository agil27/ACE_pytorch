import pandas
from sklearn.cluster import AgglomerativeClustering
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import torch

from torchvision import models
import torch.nn as nn


class cluster(object):
    def __init__(self):
        self.CLUSTER_K = 4
        self.input_size = [24,24]
        self.batch_size = 32
        self.activation_feature = None
    def resnet_feature(self):

        def hook(model, input, output):
            self.activation_feature = output.detach()

        model_resnet = models.resnet18(pretrained=True)
        model_resnet.avgpool.register_forward_hook(hook)
        return model_resnet

    def data_loader(self, file_name):
        data_transforms = transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        image_dataset_train = datasets.ImageFolder(file_name, data_transforms)
        train_loader = DataLoader(image_dataset_train, batch_size=self.batch_size)
        return train_loader
    def __call__(self, file_name):
        train_loader = self.data_loader(file_name)
        model = self.resnet_feature()
        data_feature = np.ndarray(shape=[0,512])
        for inputs, labels in train_loader:
            model(inputs)
            features = torch.flatten(self.activation_feature,1).numpy()
            data_feature = np.vstack((data_feature, features))

        aggModel = AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto',
                                    connectivity=None, linkage='ward', memory=None, n_clusters=5)
        aggModel = aggModel.fit(data_feature)

        target_list = aggModel.labels_
        return target_list


if __name__ == '__main__':
    cluster_test = cluster()
    target_list = cluster_test('../data_test')
    print(target_list)