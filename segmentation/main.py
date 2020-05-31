from segmentation import *
from utils import image_show

if __name__ == '__main__':
  concept='bike'
  source_dir='../data/'

  # put image of one class in source_dir/concept
  dataset, image_numbers, patches = create_patches('slic',source_dir,concept)
  
  # returns: segmentations extract from the class, three level (8, 20, 30)
  # dataset: shape(num_seg, H, W, 3)  # mask on origal image
  # image_numbers: shape(num_seg), indicates which image the segmentation comes from
  # patches: shape(num_seg, H, W, 3)  # resize the masked image to model input size, default (229, 229), change the value in def _extract_patch(... image_shape=(229, 229))

  print("Segementation number: ", image_numbers.shape[0])
  # P.S. 10 image generate about 300 segmentions

  # image_show(dataset[20])
  # image_show(patches[20])