import shutil
import os
from sklearn.cluster import AgglomerativeClustering
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import torch
import argparse
from torchvision import models
from mydata import MyDataset
parser = argparse.ArgumentParser()
parser.add_argument('--src_path', default='data/seg_result')
parser.add_argument('--tgt_path', default='data/cluster_result')
parser.add_argument('--class_name', default='bike')
args = parser.parse_args()

class cluster(object):
    def __init__(self):
        self.input_size = [128,128]
        self.batch_size = 32
        self.activation_feature = None
    def resnet_feature(self):

        def hook(model, input, output):
            self.activation_feature = output.detach()

        model_resnet = models.resnet18(pretrained=True)
        model_resnet.avgpool.register_forward_hook(hook)
        return model_resnet

    def data_loader(self, base_path, class_name):
        data_transforms = transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        image_dataset_train = MyDataset(base_path, class_name, transform=data_transforms)
        train_loader = DataLoader(image_dataset_train, batch_size=self.batch_size)
        return train_loader
    def __call__(self, base_path, class_name, n_clusters):
        train_loader = self.data_loader(base_path, class_name)
        pic_path_list = train_loader.dataset.names_list

        model = self.resnet_feature()
        data_feature = np.ndarray(shape=[0,512])
        for inputs, filenames in train_loader:
            print(inputs[0].shape)
            model(inputs)
            features = torch.flatten(self.activation_feature,1).numpy()
            data_feature = np.vstack((data_feature, features))
        #print(data_feature.shape)
        aggModel = AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto',
                                    connectivity=None, linkage='ward', memory=None, n_clusters=n_clusters)
        aggModel = aggModel.fit(data_feature)

        target_list = aggModel.labels_
        return target_list, pic_path_list


def main():
    cluster_test = cluster()

    # param: path_name: the path to the segmented images, path format required: path_name/xxx/yyy.png
    # param: n_clusters: the number of clusters
    # return target_list: the list of clustering result
    # pic_path_list: the list of each picture's path
    target_list, pic_path_list = cluster_test(base_path=args.src_path, class_name=args.class_name, n_clusters=30)

    # optional:
    # save all pictures according to clustering result:
    dst_root_path = args.tgt_path + '/' + args.class_name
    for index, pic_path in enumerate(pic_path_list):
        pic_path = args.src_path + '/' + args.class_name + '/' + pic_path
        dst_path = dst_root_path + '/%02d/' % target_list[index]
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        ttt = shutil.copy(pic_path, dst_path)

if __name__ == '__main__':
    main()