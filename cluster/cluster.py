import shutil
import os
from sklearn.cluster import AgglomerativeClustering
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import torch
from torchvision import models

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
    def __call__(self, path_name, n_clusters):
        train_loader = self.data_loader(path_name)
        pic_path_list = [x[0] for x in train_loader.dataset.samples]

        model = self.resnet_feature()
        data_feature = np.ndarray(shape=[0,512])
        for inputs, labels in train_loader:
            model(inputs)
            features = torch.flatten(self.activation_feature,1).numpy()
            data_feature = np.vstack((data_feature, features))

        aggModel = AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto',
                                    connectivity=None, linkage='ward', memory=None, n_clusters=n_clusters)
        aggModel = aggModel.fit(data_feature)

        target_list = aggModel.labels_
        return target_list, pic_path_list

if __name__ == '__main__':
    cluster_test = cluster()

    # param: path_name: the path to the segmented images, path format required: path_name/xxx/yyy.png
    # param: n_clusters: the number of clusters
    # return target_list: the list of clustering result
    # pic_path_list: the list of each picture's path
    target_list, pic_path_list = cluster_test(path_name='../data_seg_result', n_clusters=30)
    print(target_list)


    # optional:
    # save all pictures according to clustering result:
    dst_root_path = 'cluster_result'
    for index, pic_path in enumerate(pic_path_list):
        dst_path = dst_root_path+'/%02d/'%target_list[index]
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        ttt = shutil.copy(pic_path, dst_path)