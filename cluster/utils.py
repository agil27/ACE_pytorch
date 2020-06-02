from segmentation.utils import *
from segmentation.segment import *
import matplotlib.pyplot as plt
from PIL import Image
import sys
if __name__ == '__main__':
    concept = 'bike'
    source_dir = '../data/'

    slic = SlicSegment(source_dir, concept)
    image_numbers, dataset, patches = slic.create_patches()

    # save the segmented pictures
    dataset = dataset.astype(np.float32)
    for i in range(dataset.shape[0]):
        image_show(dataset[i])
        plt.savefig('../data_seg_result/1/%05d.png'%i)
        print(i)



