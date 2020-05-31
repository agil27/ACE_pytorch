import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def image_show(image, nrows=1, ncols=1, cmap='gray'):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    return fig, ax

def load_image_from_file(filename, shape):
  """Given a filename, try to open the file. If failed, return None.
  Args:
    filename: location of the image file
    shape: the shape of the image file to be scaled
  Returns:
    the image if succeeds, None if fails.
  Rasies:
    exception if the image was not the right shape.
  """
#   if not tf.gfile.Exists(filename):
#     tf.logging.error('Cannot find file: {}'.format(filename))
#     return None
#   try:
  img = np.array(Image.open(filename).resize(
      shape, Image.BILINEAR))
  # Normalize pixel values to between 0 and 1.
  img = np.float32(img) / 255.0
  if not (len(img.shape) == 3 and img.shape[2] == 3):
    return None
  else:
    return img

#   except Exception as e:
#     print(e)
#     return None
  return img

def load_images_from_files(filenames, max_imgs=500, return_filenames=False,
                           do_shuffle=True,
                           # run_parallel=True,
                           shape=(299, 299),
                           #num_workers=100
                          ):
  """Return image arrays from filenames.
  Args:
    filenames: locations of image files.
    max_imgs: maximum number of images from filenames.
    return_filenames: return the succeeded filenames or not
    do_shuffle: before getting max_imgs files, shuffle the names or not
    run_parallel: get images in parallel or not
    shape: desired shape of the image
    num_workers: number of workers in parallelization.
  Returns:
    image arrays and succeeded filenames if return_filenames=True.
  """
  imgs = []
  # First shuffle a copy of the filenames.
  filenames = filenames[:]
  if do_shuffle:
    np.random.shuffle(filenames)
  if return_filenames:
    final_filenames = []
#   if run_parallel:
#     pool = multiprocessing.Pool(num_workers)
#     imgs = pool.map(lambda filename: load_image_from_file(filename, shape),
#                     filenames[:max_imgs])
#     if return_filenames:
#       final_filenames = [f for i, f in enumerate(filenames[:max_imgs])
#                          if imgs[i] is not None]
#     imgs = [img for img in imgs if img is not None]
#   else:
  for filename in filenames:
#    print(filename)
    img = load_image_from_file(filename, shape)
#    print(img)
    if img is not None:
      imgs.append(img)
      if return_filenames:
        final_filenames.append(filename)
    if len(imgs) >= max_imgs:
      break

  if return_filenames:
    return np.array(imgs), final_filenames
  else:
    return np.array(imgs)