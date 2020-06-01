from utils import *
import os
import numpy as np
import skimage.segmentation as segmentation

class SlicSegment:
  def __init__(self, source_dir, concept):
    self.source_dir = source_dir
    self.concept = concept

  def load_concept_imgs(self, max_imgs=1000, image_shape=(299, 299)):
    """Loads all colored images of a concept.
    Args:
      max_imgs: maximum number of images to be loaded
      image_shape: target resize image shape, default (299, 299)
    Returns:
      Images of the desired concept or class.
    """
    concept_dir = os.path.join(source_dir, concept)
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

  def create_patches(self, param_dict=None):
    """Creates a set of image patches using slic methods.
    Args:
      param_dict: Contains parameters of the superpixel method used in the form
                of {'param1':[a,b,...], 'param2':[z,y,x,...], ...}. For instance
                {'n_segments':[15,50,80], 'compactness':[10,10,10]}.
    """
    if param_dict is None:
      param_dict = {}
    dataset, image_numbers, patches = [], [], []
    raw_imgs = self.load_concept_imgs()
    discovery_images = raw_imgs

    for fn, img in enumerate(discovery_images):
      image_superpixels, image_patches = self._return_superpixels(
          img, param_dict)
      for superpixel, patch in zip(image_superpixels, image_patches):
        dataset.append(superpixel)
        patches.append(patch)
        image_numbers.append(fn)
    dataset, simage_numbers, patches =\
    np.array(dataset), np.array(image_numbers), np.array(patches)

    return image_numbers, dataset,  patches

  def _return_superpixels(self, img, param_dict=None):
    """Returns all patches for one image.

        Given an image, calculates superpixels for each of the parameter lists in
        param_dict and returns a set of unique superpixels by
        removing duplicates. If two patches have Jaccard similarity more than 0.5,
        they are concidered duplicates.
    Args:
      img: The input image
      param_dict: Contains parameters of the superpixel method used in the form
                of {'param1':[a,b,...], 'param2':[z,y,x,...], ...}. For instance
                {'n_segments':[8,20,30], 'compactness':[20,20,20]} for slic
                method.
    """
    if param_dict is None:
      param_dict = {}
    n_segmentss = param_dict.pop('n_segments', [8, 20, 30])
    n_params = len(n_segmentss)
    compactnesses = param_dict.pop('compactness', [20] * n_params)
    sigmas = param_dict.pop('sigma', [1.] * n_params)

    unique_masks = []
    for i in range(n_params):
      param_masks = []
      segments = segmentation.slic(
        img, n_segments=n_segmentss[i], compactness=compactnesses[i],
        sigma=sigmas[i])
      for s in range(segments.max()):
        mask = (segments == s).astype(float)
        if np.mean(mask) > 0.001:
          unique = True
          for seen_mask in unique_masks:
            jaccard = np.sum(seen_mask * mask) / np.sum((seen_mask + mask) > 0)
            if jaccard > 0.5:
              unique = False
              break
          if unique:
            param_masks.append(mask)
      unique_masks.extend(param_masks)
    superpixels, patches = [], []
    while unique_masks:
      superpixel, patch = self._extract_patch(img, unique_masks.pop())
      superpixels.append(superpixel)
      patches.append(patch)
    return superpixels, patches

  def _extract_patch(self, image, mask, average_image_value=117, image_shape=(229, 229)):
    """Extracts a patch out of an image.

    Args:
      image: The original image
      mask: The binary mask of the patch area
      average_image_value: default color of the area unmasked
    Returns:
      image_resized: The resized patch such that its boundaries touches the
        image boundaries
      patch: The original patch. Rest of the image is padded with average value
    """
    mask_expanded = np.expand_dims(mask, -1)
    patch = (mask_expanded * image + (
        1 - mask_expanded) * float(average_image_value) / 255)
    ones = np.where(mask == 1)
    h1, h2, w1, w2 = ones[0].min(), ones[0].max(), ones[1].min(), ones[1].max()
    image = Image.fromarray((patch[h1:h2, w1:w2] * 255).astype(np.uint8))
    image_resized = np.array(image.resize(image_shape,
                                          Image.BICUBIC)).astype(float) / 255
    return image_resized, patch