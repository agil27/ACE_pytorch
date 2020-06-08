# detectron2
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import numpy as np
import cv2
import os
import random
from PIL import Image

from utils import *


class DeepSegment:
    def __init__(self, source_dir, concept):
        self.source_dir = source_dir
        self.concept = concept

        cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        cfg.MODEL.DEVICE = "cpu"
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.predictor = DefaultPredictor(cfg)

    def load_concept_imgs(self, max_imgs=1000, image_shape=(299, 299)):
        """Loads all colored images of a concept.
        Args:
          max_imgs: maximum number of images to be loaded
          image_shape: target resize image shape, default (299, 299)
        Returns:
          Images of the desired concept or class.
        """
        concept_dir = os.path.join(self.source_dir, self.concept)
        img_paths = [
            os.path.join(concept_dir, d)
            for d in os.listdir(concept_dir)
        ]
        return load_images_from_files(
            img_paths,
            max_imgs=max_imgs,
            return_filenames=False,
            do_shuffle=False,
            shape=image_shape,
        )

    def create_patches(self):
        dataset, image_numbers, patches = [], [], []

        raw_imgs = self.load_concept_imgs() * 255
        raw_imgs = raw_imgs.astype(np.uint8)

        for index, img in enumerate(raw_imgs):
            outputs = self.predictor(img)
            boxes = outputs['instances'].pred_boxes.tensor.numpy()
            for box in boxes:
                img1, patch = self._extract_patch(img, box)
                dataset.append(img1)
                patches.append(patch)
                image_numbers.append(index)

        dataset, simage_numbers, patches = \
            np.array(dataset), np.array(image_numbers), np.array(patches)

        return image_numbers, dataset, patches

    def _extract_patch(self, img, box, image_shape=(229, 229)):
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        patch = Image.fromarray((img[x1:x2, y1:y2] * 255).astype(np.uint8))
        patch = np.array(patch.resize(image_shape, Image.BICUBIC)).astype(np.float) / 255
        img = img.astype(np.float) / 255
        return img, patch


if __name__ == "__main__":
    concept = 'bike'
    source_dir = '../data/'

    mask_rcnn = DeepSegment(source_dir, concept)
    image_numbers, dataset, patches = mask_rcnn.create_patches()

    i = 0
    for img, p in zip(dataset, patches):
        image_show(img)
        plt.savefig("temp/dataset-{0}".format(i))
        image_show(p)
        plt.savefig("temp/patch-{0}".format(i))
        i += 1
