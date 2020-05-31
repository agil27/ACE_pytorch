from skimage.segmentation import slic
import skimage.segmentation as segmentation
from utils import *
import os

def load_concept_imgs(source_dir, concept, max_imgs=1000, image_shape=(299, 299)):
    """Loads all colored images of a concept.

    Args:
      concept: The name of the concept to be loaded
      max_imgs: maximum number of images to be loaded

    Returns:
      Images of the desired concept or class.
    """
    concept_dir = os.path.join(source_dir, concept)
    img_paths = [
        os.path.join(concept_dir, d)
        #for d in tf.gfile.ListDirectory(concept_dir)
        for d in os.listdir(concept_dir)
    ]
    # print(img_paths)
    return load_images_from_files(
        img_paths,
        max_imgs=max_imgs,
        return_filenames=False,
        do_shuffle=False,
        # run_parallel=(self.num_workers > 0),
        shape=image_shape,
        #num_workers=self.num_workers
    )

def create_patches(method,source_dir,concept,param_dict=None):
  """Creates a set of image patches using superpixel methods.

  This method takes in the concept discovery images and transforms it to a
  dataset made of the patches of those images.

  Args:
    method: The superpixel method used for creating image patches. One of
      'slic', 'watershed', 'quickshift', 'felzenszwalb'.
    discovery_images: Images used for creating patches. If None, the images in
      the target class folder are used.

    param_dict: Contains parameters of the superpixel method used in the form
              of {'param1':[a,b,...], 'param2':[z,y,x,...], ...}. For instance
              {'n_segments':[15,50,80], 'compactness':[10,10,10]} for slic
              method.
  """
  if param_dict is None:
    param_dict = {}
  dataset, image_numbers, patches = [], [], []
#if discovery_images is None:
  raw_imgs = load_concept_imgs(source_dir, concept)
  discovery_images = raw_imgs
  
#  return discovery_images
#     else:
#       self.discovery_images = discovery_images
#     if self.num_workers:
#       pool = multiprocessing.Pool(self.num_workers)
#       outputs = pool.map(
#           lambda img: self._return_superpixels(img, method, param_dict),
#           self.discovery_images)
#       for fn, sp_outputs in enumerate(outputs):
#         image_superpixels, image_patches = sp_outputs
#         for superpixel, patch in zip(image_superpixels, image_patches):
#           dataset.append(superpixel)
#           patches.append(patch)
#           image_numbers.append(fn)
#     else:
  for fn, img in enumerate(discovery_images):
    image_superpixels, image_patches = _return_superpixels(
        img, method, param_dict)
    for superpixel, patch in zip(image_superpixels, image_patches):
      dataset.append(superpixel)
      patches.append(patch)
      image_numbers.append(fn)
  dataset, simage_numbers, patches =\
  np.array(dataset), np.array(image_numbers), np.array(patches)
  
  return dataset, simage_numbers, patches

def _return_superpixels(img, method, param_dict=None):
    """Returns all patches for one image.

    Given an image, calculates superpixels for each of the parameter lists in
    param_dict and returns a set of unique superpixels by
    removing duplicates. If two patches have Jaccard similarity more than 0.5,
    they are concidered duplicates.

    Args:
      img: The input image
      method: superpixel method, one of slic, watershed, quichsift, or
        felzenszwalb
      param_dict: Contains parameters of the superpixel method used in the form
                of {'param1':[a,b,...], 'param2':[z,y,x,...], ...}. For instance
                {'n_segments':[15,50,80], 'compactness':[10,10,10]} for slic
                method.
    Raises:
      ValueError: if the segementation method is invaled.
    """
    if param_dict is None:
      param_dict = {}
#     if method == 'slic':
    n_segmentss = param_dict.pop('n_segments', [8, 20, 30])
    n_params = len(n_segmentss)
    compactnesses = param_dict.pop('compactness', [20] * n_params)
    sigmas = param_dict.pop('sigma', [1.] * n_params)
#     elif method == 'watershed':
#       markerss = param_dict.pop('marker', [15, 50, 80])
#       n_params = len(markerss)
#       compactnesses = param_dict.pop('compactness', [0.] * n_params)
#     elif method == 'quickshift':
#       max_dists = param_dict.pop('max_dist', [20, 15, 10])
#       n_params = len(max_dists)
#       ratios = param_dict.pop('ratio', [1.0] * n_params)
#       kernel_sizes = param_dict.pop('kernel_size', [10] * n_params)
#     elif method == 'felzenszwalb':
#       scales = param_dict.pop('scale', [1200, 500, 250])
#       n_params = len(scales)
#       sigmas = param_dict.pop('sigma', [0.8] * n_params)
#       min_sizes = param_dict.pop('min_size', [20] * n_params)
#     else:
#       raise ValueError('Invalid superpixel method!')
    unique_masks = []
    for i in range(n_params):
      param_masks = []
#       if method == 'slic':
      segments = segmentation.slic(
        img, n_segments=n_segmentss[i], compactness=compactnesses[i],
        sigma=sigmas[i])
#       elif method == 'watershed':
#         segments = segmentation.watershed(
#             img, markers=markerss[i], compactness=compactnesses[i])
#       elif method == 'quickshift':
#         segments = segmentation.quickshift(
#             img, kernel_size=kernel_sizes[i], max_dist=max_dists[i],
#             ratio=ratios[i])
#       elif method == 'felzenszwalb':
#         segments = segmentation.felzenszwalb(
#             img, scale=scales[i], sigma=sigmas[i], min_size=min_sizes[i])
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
      superpixel, patch = _extract_patch(img, unique_masks.pop())
      superpixels.append(superpixel)
      patches.append(patch)
    return superpixels, patches


def _extract_patch(image, mask, average_image_value=117, image_shape=(229, 229)):
    """Extracts a patch out of an image.

    Args:
      image: The original image
      mask: The binary mask of the patch area

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